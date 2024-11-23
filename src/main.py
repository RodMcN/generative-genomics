
# download -> preprocess -> PCA -> XGB -> VAE -> XGB -> XGB
import tomllib
from pathlib import Path
from typing import Union, Dict
from argparse import ArgumentParser
from data.preprocess import preprocess
from models.xgb import XGBWrapper
from copy import copy
from models.pca import PCAWrapper
import pickle
from utils import get_logger, StashableVar, clear_gpu_memory
from models.vae_train import train_vae as train_vae_
from models.vae import CVAE

logger = get_logger(__name__) 

class Pipeline:
    def __init__(self, config: Union[Dict, str, Path]):
        if isinstance(config, (str, Path)):
            with open(config, 'rb') as f:
                conf = tomllib.load(f)
        
        self.config = conf
        self.dataset_name = self.config["dataset"]["name"]
        
        self.output_dir = Path(conf["io"]["data_dir"])
        if not self.output_dir.exists():
            logger.warning(f"Output directory {self.output_dir} does not exist. Creating it.")
            self.output_dir.mkdir(parents=True)
    
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
        
        self.train_X = StashableVar(train_X)
        self.train_y = StashableVar(train_y)
        self.val_X = StashableVar(val_X).stash()
        self.val_y = StashableVar(val_y)
        # stash the test data for later
        self.test_X = StashableVar(test_X).stash()
        self.test_y = StashableVar(test_y).stash()
        self.categories = StashableVar(categories)


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
        xgb = XGBWrapper(xgb_params, name=model_name)
        xgb.fit(train_X, train_y, val_X, val_y)
        
        out_file = self.output_dir / f"{self.dataset_name}_{model_name}.xgb"
        logger.info(f"Saving xgb model to {out_file}")
        xgb.save(out_file)
        return xgb
        
    
    def eval_xgb(self, model, test_X, test_y, categories, category_mapping=None):
        logger.info("Evaluating XGB")
        if isinstance(test_X, (str, Path)):
            with open(test_X, "rb") as f:
                test_X = pickle.load(f)
        if isinstance(test_y, (str, Path)):
            with open(test_y, "rb") as f:
                test_y = pickle.load(f)
        
        results = model.evaluate(test_X, test_y, categories, category_mapping)
        out_file = self.output_dir / f"{self.dataset_name}_{model.name}_xgb_.tsv"
        logger.info(f"Saving results to {out_file}")
        results.to_csv(out_file, sep="\t")
        logger.info(results)
        
        
    def shap_explain(self):
        pass
    
    def train_vae(self):
        logger.info("Training VAE")
        config = self.config["vae"]
        model_path = self.output_dir / f"{self.dataset_name}_vae.pt"
        train_vae_(self.X_train.load(), 
                   self.y_train.load(),
                   self.X_val.load(),
                   self.y_val.load(),
                   config,
                   model_path)
        return model_path
    
    def train_vxgb(self, vae_path):
        # train on VAE generated data
        # generate 2k train samples and 100 eval samples per cell type
        vae = CVAE.load(vae_path, device="cuda")
        train_y = [i for i in range(len(self.categories.load())) for _ in range(2000)]
        train_X = vae.generate(train_y, seed=42).cpu().numpy()
        
        val_y = [i for i in range(len(self.categories.load())) for _ in range(100)]
        val_X = vae.generate(val_y, seed=1337).cpu().numpy()
        
        xgb_vae = self.train_xgb(train_X, train_y, val_X, val_y, "synthetic_data")
        return xgb_vae

    
    def run(self):
        data_file = self.download()
        
        self.split_data(self.preprocess_data(data_file))
        
        # train_X, etc are initialised in split_data
        train_X = self.fit_pca(self.train_X.load_once())
        # original X not needed after pca
        self.train_X = None
        val_X = self.apply_pca(self.val_X.load_once())
        self.val_X = None
        clear_gpu_memory()
        
        # train XGBoost on the "real" data
        train_y = self.train_y.load()
        val_y = self.val_y.load()
        xgb_real = self.train_xgb(train_X, train_y, val_X, val_y, "real_data")
        
        # eval xgboost
        categories = self.categories.load()
        self.eval_xgb(xgb_real, self.test_X.load_once(), self.test_y.load_once(), categories, category_mapping=None)
        self.shap_explain()
        clear_gpu_memory()
        
        # train VAE
        vae_path = self.train_vae()
        
        # train and val data no longer needed
        del train_X, val_X, train_y, val_y
        clear_gpu_memory()
        
        # train XGBoost on VAE generated data
        xgb_vae = self.train_vxgb(vae_path)
        self.eval_xgb(xgb_vae, self.test_X.load_once(), self.test_y.load_once(), categories, category_mapping=None)
        self.shap_explain()
        clear_gpu_memory()
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf.toml")
    args = parser.parse_args()
    
    pipeline = Pipeline(args.config)
    
    logger.info("Starting pipeline")
    pipeline.run()
    logger.info("Finished pipeline")