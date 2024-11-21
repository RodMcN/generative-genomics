from cuml.decomposition import IncrementalPCA
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


class IPCA:
    def __init__(self, n_components=2500, batch_size=5000, scale=True):
        self.n_components = n_components
        self.batch_size = batch_size
        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
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
        
        
        X = subsampled_data.X
        # apply scaling
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # shuffle indices to reduce bias
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        # fit PCA iteratively
        total_batches = math.ceil(len(X) / self.batch_size)
        for i in tqdm(range(total_batches), desc="Computing incremental PCA"):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(X))
            batch_indices = indices[start_idx:end_idx]
            if len(batch_indices) >= self.n_components:
                self.pca.partial_fit(X[batch_indices])
            else:
                logger.warning("Last batch size is smaller than n_components. Stopping.")
        self.is_fitted = True
    
    
    def transform(self, data):
        if hasattr(data, "X"):
            data = data.X
        if self.scaler is not None:
            data = self.scaler.transform(data)
        return self.pca.transform(data)
    
    def save(self, path: str):
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'n_components': self.n_components,
            'batch_size': self.batch_size
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(n_components=data['n_components'], batch_size=data['batch_size'])
        obj.scaler = data['scaler']
        obj.pca = data['pca']
        return obj
