
# download -> preprocess -> PCA -> XGB -> VAE -> XGB -> XGB
import tomllib
from pathlib import Path
from typing import Union, Dict
from argparse import ArgumentParser
from data.preprocess import preprocess
from models.xgb import XGBWrapper
from copy import copy
from models.pca import PCAWrapper
import logging 
import os
import pickle

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Union[Dict, str, Path]):
        if isinstance(config, (str, Path)):
            with open(config, 'rb') as f:
                conf = tomllib.load(f)
        
        self.config = conf
        self.dataset_name = self.config["dataset"]["name"]
    
    def download(self) -> Path:
        dataset_name = self.config["dataset"]["name"]
        output_file = Path(f"/workdir/{dataset_name}.project.h5ad")
        if output_file.exists():
            logger.info(f"{output_file} already exists")
            return output_file
    
    def preprocess_data(self, dataset_file: Path):
        logger.info(f"Preprocessing {dataset_file}")
        dataset_config = self.config["dataset"]
        dataset_config = copy(dataset_config)
        dataset_config.pop("name")
        data = preprocess(dataset_file, **dataset_config)
        return data
    
    def split_data(self, data):
        logger.info("Splitting data")
        train_data = data[data.obs['split'] == "train"]
        train_X = train_data.X.toarray()
        train_y = train_data.obs['celltype_annotation'].cat.codes.values
        
        val_data = data[data.obs['split'] == "val"]
        val_X = val_data.X.toarray()
        val_y = val_data.obs['celltype_annotation'].cat.codes.values
        
        test_data = data[data.obs['split'] == "test"]
        test_X = test_data.X.toarray()
        test_y = test_data.obs['celltype_annotation'].cat.codes.values
        
        categories = data.obs['celltype_annotation'].cat.categories
        
        return train_X, train_y, val_X, val_y, test_X, test_y, categories
        
    
    def fit_pca(self, data):
        logger.info("Fitting PCA")
        pca_config = self.config["pca"]
        self.pca = PCAWrapper(**pca_config)
        return self.pca.fit_transform(data)
    
    def apply_pca(self, data):
        logger.info("Applying PCA")
        return self.pca.transform(data)
    
    def inverse_pca(self):
        pass
    
    def train_xgb(self, train_X, train_y, val_X, val_y, model_name):
        logger.info("Training XGB")
        xgb_params = self.config["xgboost"]
        xgb = XGBWrapper(**xgb_params)
        xgb.fit(train_X, train_y, val_X, val_y)
        
        out_file = Path(f"/workdir/{self.dataset_name}_{model_name}.xgb")
        xgb.save(out_file)
        return xgb
        
    
    def eval_xgb(self, model, test_X, test_y):
        pass
    
    def shap_explain(self):
        pass
    
    def train_vae(self):
        pass
    
    def run(self):
        data_file = self.download()
        
        data = self.split_data(self.preprocess_data(data_file))
        train_X, train_y, val_X, val_y, test_X, test_y, categories = data
        
        # stash the test data for later
        with open("test_X.pkl", "wb") as f:
            pickle.dump(test_X, f)
        with open("test_y.pkl", "wb") as f:
            pickle.dump(test_y, f)
        del test_X, test_y
        
        
        train_X = self.fit_pca(train_X)
        val_X = self.apply_pca(val_X)
        
        # train XGBoost on the "real" data
        self.train_xgb(train_X, train_y, val_X, val_y, "real_data")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf.toml")
    args = parser.parse_args()
    
    pipeline = Pipeline(args.config)
    pipeline.run()