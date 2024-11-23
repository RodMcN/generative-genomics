import warnings
try:
    from cuml.decomposition import IncrementalPCA as IPCA_GPU, PCA as PCA_GPU
    HAS_CUML = True
except ModuleNotFoundError:
    warnings.warn("cuML is not installed. Falling back to CPU PCA.")
    HAS_CUML = False
from sklearn.decomposition import IncrementalPCA as IPCA_CPU, PCA as PCA_CPU
    
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import logging
import os
import math
import joblib

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


class PCAWrapper:
    def __init__(self, n_components=2500, scale=True, gpu=True, incremental=True, batch_size=5000, random_state=42):
        self.n_components = n_components
        self.batch_size = batch_size
        self.gpu = gpu
        self.incremental = incremental
        self.is_fitted = False
        self.random_state = random_state

        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        
        if gpu:
            if not HAS_CUML:
                logger.warning("cuML is not installed. Falling back to CPU PCA.")
                gpu = False

        if incremental:
            PCA_Class = IPCA_GPU if gpu else IPCA_CPU
            self.pca = PCA_Class(n_components=n_components, batch_size=batch_size, random_state=random_state)
        else:
            PCA_Class = PCA_GPU if gpu else PCA_CPU
            self.pca = PCA_Class(n_components=n_components, random_state=random_state)
            self.batch_size = None
        logger.debug(f"Using {PCA_Class.__name__}")
            
    
    def fit_subsample(self, data, class_col, max_per_class=5_000):
        subsampled_data = []
        class_counts = Counter(data.obs[class_col])

        # sample up to max_per_class for each class 
        for cls, count in tqdm(class_counts.items(), desc="Subsampling data"):
            class_subset = data[data.obs[class_col] == cls]
            
            if count > max_per_class:
                sampled_indices = class_subset.obs.sample(max_per_class, random_state=42).index
                class_subset = class_subset[sampled_indices]
            
            subsampled_data.append(class_subset)

        subsampled_data = subsampled_data[0].concatenate(
            *subsampled_data[1:], 
            join='outer',
            batch_key=None
        )
        logger.info(f"Subsampled {data.shape} to {subsampled_data.shape}")
        
        if self.incremental:
            self.fit_incremental(subsampled_data.X)
        else:
            self.fit(subsampled_data.X)

    
    def fit_incremental(self, data):
        # apply scaling
        if self.scaler is not None:
            logger.debug(f"Calling scaler.fit_transform({data.shape})")
            data = self.scaler.fit_transform(data)
        
        # shuffle indices to reduce bias
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        
        # fit PCA iteratively
        total_batches = math.ceil(len(data) / self.batch_size)
        for i in tqdm(range(total_batches), desc="Computing incremental PCA"):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(data))
            batch_indices = indices[start_idx:end_idx]
            if len(batch_indices) >= self.n_components:
                self.pca.partial_fit(data[batch_indices])
            else:
                logger.warning("Last batch size is smaller than n_components. Stopping.")
        self.is_fitted = True
    
    
    def fit(self, data):
        if self.incremental:
            logger.debug(f"Calling fit_incremental({data.shape})")
            # fit_incremental will call scaler.fit_transform
            self.fit_incremental(data)
        else:
            if self.scaler is not None:
                logger.debug(f"Calling scaler.fit_transform({data.shape})")
                data = self.scaler.fit_transform(data)
            logger.debug(f"Calling pca.fit({data.shape})")
            self.pca.fit(data)
            self.is_fitted = True
    
    
    def transform(self, data):
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet.")
        if hasattr(data, "X"):
            data = data.X
        if self.scaler is not None:
            logger.debug(f"Calling scaler.transform({data.shape})")
            data = self.scaler.transform(data)
        logger.debug(f"Calling pca.transform({data.shape})")
        return self.pca.transform(data)

    
    def fit_transform(self, data):
        if not self.incremental:
            logger.debug(f"Calling fit_transform({data.shape})")
            data = self.scaler.fit_transform(data)
            logger.debug(f"Calling pca.fit_transform({data.shape})")
            data = self.pca.fit_transform(data)
            self.is_fitted = True
            return data
        else:
            logger.debug(f"Calling fit_incremental({data.shape})")
            self.fit_incremental(data)
            logger.debug(f"Calling transform({data.shape})")
            return self.transform(data)
    
    
    def save(self, path: str):
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'n_components': self.n_components,
            'batch_size': self.batch_size,
            'gpu': self.gpu,
            'incremental': self.incremental,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }, path)


    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(
            n_components=data['n_components'],
            batch_size=data['batch_size'],
            gpu=data['gpu'],
            incremental=data['incremental'],
            random_state=data['random_state']
        )
        obj.scaler = data['scaler']
        obj.pca = data['pca']
        obj.is_fitted = data['is_fitted']
        return obj
