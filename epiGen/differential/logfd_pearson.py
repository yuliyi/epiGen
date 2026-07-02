import scanpy as sc
import episcanpy as epi
import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy
import pickle
from anndata import AnnData
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix

def tfidf_seurat(adata: ad.AnnData, scale_factor=10000) -> None:
    r"""
    TF-IDF normalization.
    This function normalizes the ATAC peak counts.

    Parameters
    ----------
    adata : ad.AnnData
        Input AnnData object containing raw ATAC counts in `.X`.

    Returns
    -------
    X_tfidf : scipy.sparse matrix or np.ndarray
        TF-IDF normalized matrix.
    """
    X = adata.X
    
    # 1. Term Frequency (TF)
    if scipy.sparse.issparse(X):
        row_sums = X.sum(axis=1).A1
        row_sums[row_sums == 0] = 1
        tf = X.multiply(scale_factor / row_sums.reshape(-1, 1))
    else:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        tf = X * (scale_factor / row_sums)
    
    # 2. Inverse Document Frequency (IDF)
    n_cells = X.shape[0]
    
    if scipy.sparse.issparse(X):
        cells_per_feature = X.getnnz(axis=0)
        idf = n_cells / (cells_per_feature + 1)
    else:
        cells_per_feature = np.count_nonzero(X, axis=0)
        idf = n_cells / (cells_per_feature + 1)
    
    # 3. TF-IDF Calculation with log1p transformation
    if scipy.sparse.issparse(tf):
        X_tfidf = tf.multiply(idf)
        X_tfidf.data = np.log1p(X_tfidf.data)  # log(1+x)
        if hasattr(X_tfidf, 'tocsr'):
            X_tfidf = X_tfidf.tocsr()
    else:
        X_tfidf = tf * idf
        X_tfidf = np.log1p(X_tfidf)
    return X_tfidf

# ---------------------------------------------------------
# Load and preprocess Ground Truth ATAC data
# ---------------------------------------------------------
prefix = "/path"
atac_truth = ad.read_h5ad(prefix + f"/atac_data/atac_independence_GSE234521.h5ad")
atac_obs = atac_truth.obs_names
test_idx = np.load(os.path.join(prefix, 'test_idx_GSE234521.npy'))
celltype = pd.read_csv(prefix + '/statistic/celltype_GSE234521.csv', index_col=0)
atac_truth.obs = celltype
atac_truth = atac_truth[test_idx]
atac_truth = atac_truth[~atac_truth.obs.celltype.isin(['Unknown'])]
epi.pp.filter_features(atac_truth, min_cells=1000)
epi.pp.filter_cells(atac_truth, min_features=600)
atac_truth.X = tfidf_seurat(atac_truth)

# ---------------------------------------------------------
# Identify Ground Truth Marker Peaks
# ---------------------------------------------------------
print('*************************** ground truth')
atac_truth_filter = atac_truth.copy()
sc.tl.rank_genes_groups(atac_truth_filter, groupby='celltype', method='wilcoxon')
# Extract ranking results into pandas DataFrames
gene_names = pd.DataFrame(atac_truth_filter.uns["rank_genes_groups"]["names"])
pvals = pd.DataFrame(atac_truth_filter.uns["rank_genes_groups"]["pvals_adj"])
log_fd = pd.DataFrame(atac_truth_filter.uns["rank_genes_groups"]["logfoldchanges"])
gene_names = gene_names[(log_fd > 0.5) & (pvals < 0.05)]
# Safely drop cell types that have 1 or fewer marker peaks
valid_columns = []
for ct in gene_names.columns:
    le = gene_names[ct].dropna().count()
    print(f'ct:{ct}, ', le)
    if le > 1:
        valid_columns.append(ct)
gene_names = gene_names[valid_columns]
celltype_order = gene_names.sort_index(axis=1).columns.tolist()
# Dictionaries to store target peaks and their corresponding true logFCs
truth_peaks = {}
truth_lfcs = {}
print(f"\n*************************** Ground Truth")
for ct in celltype_order:
    valid_idx = gene_names[ct].dropna().index
    if len(valid_idx) <= 1:
        continue
        
    peaks = gene_names.loc[valid_idx, ct].values
    lfcs = log_fd.loc[valid_idx, ct].values
    
    temp_df = pd.DataFrame({'peak': peaks, 'logfc': lfcs})
    
    truth_peaks[ct] = temp_df['peak'].values
    truth_lfcs[ct] = temp_df['logfc'].values
    
    print(f"Celltype: {ct:<20} | reserve Peaks: {len(temp_df)}")

# ---------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------
def evaluate_prediction(pred_adata, truth_adata, truth_peaks, truth_lfcs, method_name):
    """
    Evaluate predicted ATAC profiles by comparing their logFC correlation 
    against the ground truth ATAC marker peaks.
    """
    print(f'\n*************************** Evaluating: {method_name}')
    
    pred_adata.obs_names = atac_obs[test_idx]
    pred_adata = pred_adata[truth_adata.obs_names, truth_adata.var_names].copy()
    pred_adata.X = tfidf_seurat(pred_adata)
    
    pred_adata.obs['celltype'] = truth_adata.obs['celltype'].values.tolist()
    if 'ct_a' in truth_adata.obs:
        pred_adata.obs['ct_a'] = truth_adata.obs['ct_a'].values.tolist()
        
    sc.tl.rank_genes_groups(pred_adata, groupby='celltype', method='wilcoxon')
    
    pred_names_df = pd.DataFrame(pred_adata.uns["rank_genes_groups"]["names"])
    pred_logfc_df = pd.DataFrame(pred_adata.uns["rank_genes_groups"]["logfoldchanges"])

    for mct in truth_peaks.keys():
        t_peaks = truth_peaks[mct]
        t_lfc = truth_lfcs[mct]
        # Build a dictionary to rapidly map predicted logFCs by peak name
        p_peaks = pred_names_df[mct].values
        p_lfc = pred_logfc_df[mct].values
        pred_lfc_dict = dict(zip(p_peaks, p_lfc))
        
        lfc_t_list = []
        lfc_p_list = []
        # Pair the True logFC with the Predicted logFC for each identified marker peak
        for i, peak in enumerate(t_peaks):
            lfc_t_list.append(t_lfc[i])
            lfc_p_list.append(pred_lfc_dict.get(peak, 0.0))
        # Calculate Pearson Correlation
        if len(lfc_t_list) > 1:
            rr, _ = pearsonr(lfc_t_list, lfc_p_list)
            print(f"celltype {mct:<20} logfd pearson: {rr:>7.4f}  (base on {len(lfc_t_list)} Peaks)")
        else:
            print(f"celltype {mct} have only one peak for calculate Pearson.")

# ---------------------------------------------------------
# Execute Evaluations across Models
# ---------------------------------------------------------
# scratch
atac_scratch = ad.read_h5ad(prefix + f"/prediction/allchr_GSE234521_bins.h5ad")
evaluate_prediction(atac_scratch, atac_truth, truth_peaks, truth_lfcs, "scratch")

# nofinetune
atac_noff = ad.read_h5ad(prefix + f"/prediction/allchr_GSE234521_bins_nofinetune.h5ad")
evaluate_prediction(atac_noff, atac_truth, truth_peaks, truth_lfcs, "nofinetune")

# finetune 0.1
atac_ff = ad.read_h5ad(prefix + f"/prediction/allchr_GSE234521_bins_finetune_0.1_0.8.h5ad")
evaluate_prediction(atac_ff, atac_truth, truth_peaks, truth_lfcs, "finetune 0.1_0.8")

# MIDAS 
midas_infer = ad.read_h5ad(prefix + f"/midas/guangshi_atlas_GSE234521/atlas_b_pred_GSE234521.h5ad")
evaluate_prediction(midas_infer, atac_truth, truth_peaks, truth_lfcs, "MIDAS")