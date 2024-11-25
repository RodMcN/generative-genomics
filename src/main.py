import tomllib
from pathlib import Path
from typing import Union, Dict
from argparse import ArgumentParser
from data.preprocess import preprocess
from models.xgb import XGBWrapper
from copy import copy
from models.pca import PCAWrapper
from utils import get_logger, StashableVar, clear_gpu_memory
from models.vae_train import train as train_vae_
from models.vae import CVAE
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import json
import numpy as np

sns.set_theme()

logger = get_logger(__name__)


class Pipeline:
    def __init__(self, config: Union[Dict, str, Path]):
        if isinstance(config, (str, Path)):
            with open(config, "rb") as f:
                conf = tomllib.load(f)

        self.config = conf
        self.dataset_name = self.config["dataset"]["name"]

        self.output_dir = Path(conf["io"]["data_dir"])
        if not self.output_dir.exists():
            logger.warning(
                f"Output directory {self.output_dir} does not exist. Creating it."
            )
            self.output_dir.mkdir(parents=True)
        else:
            logger.info(f"Output directory is {self.output_dir}.")
        if any(self.output_dir.iterdir()):
            logger.warning(
                f"Output directory {self.output_dir} is not empty. Data will be overwritten."
            )

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
        train_data = data[data.obs["split"] == "train"]
        train_X = train_data.X.toarray()
        train_y = train_data.obs["celltype_annotation"].cat.codes.values

        val_data = data[data.obs["split"] == "val"]
        val_X = val_data.X.toarray()
        val_y = val_data.obs["celltype_annotation"].cat.codes.values

        test_data = data[data.obs["split"] == "test"]
        test_X = test_data.X.toarray()
        test_y = test_data.obs["celltype_annotation"].cat.codes.values

        categories = data.obs["celltype_annotation"].cat.categories.values

        self.train_X = StashableVar(train_X)
        self.train_y = StashableVar(train_y)
        self.val_X = StashableVar(val_X).stash()
        self.val_y = StashableVar(val_y)
        # stash the test data for later
        self.test_X = StashableVar(test_X).stash()
        self.test_y = StashableVar(test_y).stash()
        self.categories = categories

    def fit_pca(self, data):
        logger.info("Fitting PCA")
        pca_config = self.config["pca"]
        self.pca = PCAWrapper(**pca_config)
        self.pca.fit(data)

        explained_var = self.pca.pca.explained_variance_ratio_
        explained_var = np.cumsum(explained_var)
        plt.figure()
        sns.lineplot(x=range(1, len(explained_var) + 1), y=explained_var)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.savefig(self.output_dir / f"{self.dataset_name}_pca.png")
        plt.close()

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

    def eval_xgb(self, model: "XGBWrapper"):
        logger.info("Evaluating XGB")
        results = model.evaluate(
            self.test_X.load(), self.test_y.load(), self.categories
        )
        out_file = (
            self.output_dir / f"{self.dataset_name}_{model.name}_real_eval_xgb.tsv"
        )
        logger.info(f"Saving results to {out_file}")
        results.to_csv(out_file, sep="\t")
        logger.info(results)

        # mapping = self.config['dataset'].get("category_mapping")
        mapping = "data/celltype_mapping.json"
        if mapping is not None:
            mapping = Path(mapping)
            if not mapping.exists():
                logger.warning(f"Category mapping {mapping} does not exist")
                return
            with open(mapping, "r") as f:
                mapping = json.load(f)
            results = model.evaluate(
                self.test_X.load(),
                self.test_y.load(),
                self.categories,
                category_mapping=mapping,
            )
            out_file = (
                self.output_dir
                / f"{self.dataset_name}_{model.name}_real_eval_mapped_xgb.tsv"
            )
            logger.info(f"Saving results to {out_file}")
            results.to_csv(out_file, sep="\t")
            logger.info(results)

    def eval_xgb_on_synthetic(self, model: "XGBWrapper", vae_path: Union[str, Path]):
        logger.info("Evaluating XGB")

        vae = CVAE.load(vae_path, device="cuda")
        train_y = [i for i in range(len(self.categories)) for _ in range(1000)]
        train_X = vae.generate(train_y, seed=1337).cpu().numpy()

        results = model.evaluate(train_X, train_y, self.categories)
        out_file = (
            self.output_dir / f"{self.dataset_name}_{model.name}_synth_eval_xgb.tsv"
        )
        logger.info(f"Saving results to {out_file}")
        results.to_csv(out_file, sep="\t")

        mapping = "data/celltype_mapping.json"
        if mapping is not None:
            mapping = Path(mapping)
            if not mapping.exists():
                logger.warning(f"Category mapping {mapping} does not exist")
                return
            with open(mapping, "r") as f:
                mapping = json.load(f)
            results = model.evaluate(
                train_X, train_y, self.categories, category_mapping=mapping
            )
            out_file = (
                self.output_dir
                / f"{self.dataset_name}_{model.name}_synth_eval_mapped_xgb.tsv"
            )
            logger.info(f"Saving results to {out_file}")
            results.to_csv(out_file, sep="\t")
            logger.info(results)

    def shap_explain(self, model, cell_type, filename):
        logger.info(f"Explaining {cell_type}")
        cell_type_index = self.categories.tolist().index(cell_type)
        test_y = self.test_y.load_once()
        test_X = self.test_X.load_once()
        indices_of_interest = [
            i for i, cell_type in enumerate(test_y) if cell_type == cell_type_index
        ]

        test_X = test_X[indices_of_interest]

        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(test_X)
        shap_values = shap_values[..., cell_type_index]

        shap.summary_plot(
            shap_values,
            test_X,
            feature_names=[f"PC{i+1}" for i in range(test_X.shape[-1])],
            max_display=10,
        )
        out_file = self.output_dir / f"{filename}.png"
        plt.savefig(out_file)
        plt.close()

    def train_vae(self):
        logger.info("Training VAE")
        config = self.config["vae"]
        model_path = self.output_dir / f"{self.dataset_name}_vae.pt"
        train_vae_(
            self.train_X.load_once(),
            self.train_y.load_once(),
            self.val_X.load_once(),
            self.val_y.load_once(),
            config,
            model_path,
        )
        return model_path

    def train_vxgb(self, vae_path):
        # train on VAE generated data
        # generate 2k train samples and 100 eval samples per cell type
        vae = CVAE.load(vae_path, device="cuda")
        train_y = [i for i in range(len(self.categories)) for _ in range(2000)]
        train_X = vae.generate(train_y, seed=42).cpu().numpy()

        val_y = [i for i in range(len(self.categories)) for _ in range(100)]
        val_X = vae.generate(val_y, seed=1337).cpu().numpy()

        xgb_vae = self.train_xgb(train_X, train_y, val_X, val_y, "synthetic_data")
        return xgb_vae

    def run(self):
        data_file = self.download()

        self.split_data(self.preprocess_data(data_file))

        # train_X, etc are initialised in split_data
        # original X not needed after pca
        self.fit_pca(self.train_X.load())
        self.train_X = StashableVar(self.apply_pca(self.train_X.load()))
        self.val_X = StashableVar(self.apply_pca(self.val_X.load_once()))
        self.test_X = StashableVar(self.apply_pca(self.test_X.load_once())).stash()
        clear_gpu_memory()

        # train XGBoost on the "real" data
        xgb_real = self.train_xgb(
            self.train_X.load(),
            self.train_y.load(),
            self.val_X.load(),
            self.val_y.load(),
            "real_data",
        )

        # eval xgboost
        self.eval_xgb(xgb_real)
        self.shap_explain(
            xgb_real, self.config["shap"]["cell_type"], f"shap_{self.dataset_name}_real"
        )
        clear_gpu_memory()

        # train VAE
        vae_path = self.train_vae()

        # train and val data no longer needed
        self.train_X = self.train_y = self.val_X = self.val_y = None
        clear_gpu_memory()

        # test real on fake

        # train XGBoost on VAE-generated data
        xgb_vae = self.train_vxgb(vae_path)
        # test VAE-XGB on real data
        self.eval_xgb(xgb_vae)
        # test real-XGB on VAE data
        self.eval_xgb_on_synthetic(xgb_real, vae_path)
        self.shap_explain(
            xgb_vae, self.config["shap"]["cell_type"], f"shap_{self.dataset_name}_synth"
        )
        clear_gpu_memory()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="conf.toml")
    args = parser.parse_args()

    pipeline = Pipeline(args.config)

    logger.info("Starting pipeline")
    pipeline.run()
    logger.info("Finished pipeline")
