
# download -> preprocess -> PCA -> XGB -> VAE -> XGB -> XGB
import tomllib
from pathlib import Path
from typing import Union, Dict
from argparse import ArgumentParser

class Pipeline:
    def __init__(self, config: Union[Dict, str, Path]):
        if isinstance(config, (str, Path)):
            with open(config, 'rb') as f:
                conf = tomllib.load(f)
        
        self.config = conf
    
    def download(self) -> Path:
        pass
    
    def preprocess(self):
        pass
    
    def train_xgb(self):
        pass
    
    def train_vae(self):
        pass
    
    def run(self):
        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="config.toml")
    args = parser.parse_args()
    
    pipeline = Pipeline(args.config)
    pipeline.run()