# Generative Genomics*
$^{\text{* technically transcriptomics}}$

Modeling Gene Expression with ML

---

This repo contains code for a small project on generating synthetic data for scRNA-Seq. Specifically, it implements a conditional variational autoencoder $^{[1]}$ to generate gene expression profiles for specific cell types.

Additionally, the repository includes code for training an XGBoost model to predict cell types based on gene expression profiles.

---

The main pipeline in implemented in `main.py`. the `Pipeline` class orchestrates data preprocessing, training an XGBoost model on real data, training a CVAE on the same data, generating synthetic data, and training another XGBoost model on the synthetic data. The pipeline also computes F1 scores for evaluating both XGBoost models.

---
### Contents
- `src/data/preprocess.py`: Contains the data preprocessing pipeline which includes cleaning and filtering of scRNA-Seq data using `scanpy` and the `Annotated data` format.

- `src/models/pca.py`: Includes a wrapper for PCA with optional GPU support using `cuML` and optional incremental fitting of large datasets.

- `src/models/vae.py`: Contains a `PyTorch` implementation of a conditional variational autoencoder

- `src/models/vae_train.py`: Contains the main training code for the CVAE.

- `src/models/vae_utils.py`: Metrics and dataset utilities for use in `vae_train.py`

- `src/models/xgb.py`: Wrapper for training and evaluating XGBClassifier

- `src/utils.py`: Assorted utilities

- `src/main.py`: Contains the training and evaluation pipeline. <b>Intended entrypoint for the codebase</b>.
---

### XGBoost accuracy example data
The following results were achieved running the pipeline on a dataset of 828,328 cells downloaded from the EBI Single Cell Expression Atlas (Experiment ID: E-ANND-5)

| Training set  | Test set  | Precision     | Recall     | F1 score     |
|---------------|-----------|--------------|--------------|--------------|
| Real data | Real data | placeholder | placeholder | placeholder |
| Real data | Synthetic data | placeholder | placeholder | placeholder |
| Synthetic data | Real data | placeholder | placeholder | placeholder |

E-ANND-5 contains 126 cell types. The results show the XGBoost models accuracy for cell type classification.

The 126 cell types were mapped to 25 different classes (the mapping can be found in `src/data/celltype_mapping.py`) and the model accuracies recalculated as follows:

| Training set  | Test set  | Precision     | Recall     | F1 score     |
|---------------|-----------|--------------|--------------|--------------|
| Real data | Real data | placeholder | placeholder | placeholder |
| Real data | Synthetic data | placeholder | placeholder | placeholder |
| Synthetic data | Real data | placeholder | placeholder | placeholder |

In this setup, if the model incorrectly predicts which of the 126 cell types, but the prediction is within the same broader class, it is counted as correct. The increased scores indicate that errors are often incorrect predictions but within the correct cell class.

These results were obtained by:
1. Downloaing and processing the E-ANND-5 dataset using `preprocess.py` function.
2. Splitting the data into a single train (70%) / test (20%) / eval (10%) split.
3. Fitting XGBoost on the training set, with early stopping using the eval set and testing on the test set.
4. Testing the fitted XGBoost on the test set, measuring precision recall and F1 score for classifying the 126 different cell types.
5. Training CVAE, conditioned on cell type, on the training set, with early stopping on the eval set.
6. Generating 1000 artificial datapoints for each of the 126 cell types with the CVAE and testing the previous XGBoost model on this dataset, using the intended cell type as the target for metrics calculation.
7. Generating 2000 artificial datapoints for each of the 126 cell types for a total of 252,000 artificial training points, and a further 100 of each cell type as an evaluation set.
8. Training XGBoost on the the artifical dataset, with early stopping.
9. Testing the new XGBoost on the test data from the real dataset E-ANND-5