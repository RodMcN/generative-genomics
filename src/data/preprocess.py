from typing import Union
from pathlib import Path
import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
from sklearn.feature_selection import VarianceThreshold


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


def process(adata_path: Union[Path, str],
                 cell_type_col: str = 'authors_cell_type',
                 doublet_col: str = 'predicted_doublet',
                 mito_col: str = 'pct_counts_mito',
                 gene_count_col: str = 'n_genes',
                 is_normalised: bool = True,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 ):
        
    
    logger.info(f"Loading {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # remove doublets
    if doublet_col:
        logger.info("Removing doublets")
        # convert to str to accomodate str and bool
        adata = adata[~(adata.obs[doublet_col].astype(str).str.lower() == 'true')]
    
    # remove nan cell type
    logger.info("Removing cells without cell type")
    adata = adata[~(adata.obs[cell_type_col].isnull())]
    
    logger.info("Splitting data")
    adata = split_data(adata, cell_type_col, test_size=test_size, val_size=val_size, random_state=random_state)
    logger.info("Filtering outliers")
    adata = filter_uncommon(adata)
    adata = iqr_filter_mito_and_genes(adata, cell_type_col=cell_type_col, mito_col=mito_col, gene_count_col=gene_count_col)
    
    if not is_normalised:
        logger.info("Normalising and transforming data")
        adata = normalise_and_transform(adata)
    
    adata = filter_low_variance(adata)
    
    return adata.copy()


def normalise_and_transform(adata, target_sum=1e4):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata

def split_data(adata, cell_type_col, test_size=0.1, val_size=0.1, random_state=42):
    # indices of all cells
    indices = np.arange(adata.n_obs)
    
    # stratify by cell type
    stratify = adata.obs[cell_type_col] if cell_type_col in adata.obs.columns else None

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # split train into train and val
    val_size_adjusted = val_size / (1 - test_size)
    _, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=adata.obs[cell_type_col].iloc[train_val_indices] if stratify is not None else None
    )

    adata.obs['split'] = 'train'
    adata.obs.loc[adata.obs.iloc[val_indices].index, "split"] = 'val'
    adata.obs.loc[adata.obs.iloc[test_indices].index, "split"] = 'test'

    return adata


def iqr_filter_mito_and_genes(adata, cell_type_col, mito_col, gene_count_col):

    if 'split' not in adata.obs.columns:
        raise ValueError("The 'split' column is missing in adata.obs. Run split_data() first.")

    filter_mask = np.ones(adata.shape[0], dtype=bool)
    
    logger.info(f"Cells before IQR filtering: {adata.shape[0]:,}")

    for cell_type in adata.obs[cell_type_col].unique():
        # indices for all cells of the current cell type
        cell_type_idx = adata.obs[cell_type_col] == cell_type

        # indices for train cells of the current cell type
        train_idx = cell_type_idx & (adata.obs['split'] == 'train')

        # subset train data for current cell type
        train_data = adata.obs.loc[train_idx]
        
        ### Mitochondrial content filtering ###
        mito_values_train = train_data[mito_col]
        Q1_mito = np.percentile(mito_values_train, 25)
        Q3_mito = np.percentile(mito_values_train, 75)
        IQR_mito = Q3_mito - Q1_mito
        upper_bound_mito = Q3_mito + 1.5 * IQR_mito
        
        ### Number of genes detected filtering ###
        gene_counts_train = train_data[gene_count_col]
        Q1_genes = np.percentile(gene_counts_train, 25)
        Q3_genes = np.percentile(gene_counts_train, 75)
        IQR_genes = Q3_genes - Q1_genes
        lower_bound_genes = Q1_genes - 1.5 * IQR_genes
        upper_bound_genes = Q3_genes + 1.5 * IQR_genes

        mito_values_all = adata.obs.loc[cell_type_idx, mito_col]
        gene_counts_all = adata.obs.loc[cell_type_idx, gene_count_col]

        # no lower bound for mitochondrial content
        mito_mask = mito_values_all <= upper_bound_mito
        genes_mask = (gene_counts_all >= lower_bound_genes) & (gene_counts_all <= upper_bound_genes)
        combined_mask = mito_mask & genes_mask

        filter_mask[cell_type_idx] = combined_mask

        num_filtered = cell_type_idx.sum() - combined_mask.sum()
        logger.debug(f"Filtered {num_filtered} cells in {cell_type}")
        
    adata = adata[filter_mask]
    logger.info(f"Cells after IQR filtering: {adata.shape[0]:,}")
    return adata


def filter_uncommon(adata, min_cells=100, min_genes=100):
    if 'split' not in adata.obs.columns:
        raise ValueError("The 'split' column is missing in adata.obs. Run split_data() first.")
    
    logger.info(f"Cells before count filtering: {adata.shape[0]:,}")
    logger.info(f"Genes before count filtering: {adata.shape[1]:,}")

    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    # Filter genes based only on the training set
    train_cells = adata.obs['split'] == 'train'
    if not train_cells.any():
        raise ValueError("No cells in the training set.")
    
    train_adata = adata[train_cells]
    gene_counts = (train_adata.X > 0).sum(axis=0)
    if hasattr(gene_counts, 'A1'):
        # .A1 converts sparse matrix to 1D array
        gene_counts = gene_counts.A1
    genes_to_keep = gene_counts >= min_cells

    adata = adata[:, genes_to_keep]
    
    logger.info(f"Cells after count filtering: {adata.shape[0]:,}")
    logger.info(f"Genes after count filtering: {adata.shape[1]:,}")
    
    return adata


def filter_low_variance(adata, threshold=0.01):    
    logger.info(f"Genes before variance filtering: {adata.shape[1]}")
    
    selector = VarianceThreshold(threshold=threshold)
    adata_norm_filtered = adata[:, selector.fit(adata.X).get_support()]
    
    logger.info(f"Genes after variance filtering: {adata_norm_filtered.shape[1]}")
    
    return adata[:, adata_norm_filtered.var_names]
