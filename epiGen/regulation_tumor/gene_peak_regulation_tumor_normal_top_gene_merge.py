import argparse
import os
import anndata as ad
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import episcanpy as epi
import copy
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from pyranges import read_gtf
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score
from utils import parse_cal_label_args, tfidf_seurat, get_genes_from_gencode, calculate_celltype_specific_correlation_with_labels
import warnings
warnings.filterwarnings('ignore')


def calculate_delta_aupr_for_celltype(delta_adata, delta_adata_permutation,
                                      labels, gene_name, gene_chrom, gene_start, gene_end,
                                      ct='Tumor', distances=[5e5, 3e5, 1e5],
                                      bin_size=10000, use_absolute_delta=True):
    """
    Evaluate the prediction performance (AUPR, AUROC) of perturbation delta 
    against Ground Truth labels across different proximity windows.
    """
    
    # Aggregate perturbations across cells
    if use_absolute_delta:
        delta_values = np.abs(delta_adata.X).mean(axis=0)
        delta_values_permutation = np.abs(delta_adata_permutation.X).mean(axis=0)
    else:
        delta_values = delta_adata.X.mean(axis=0)
        delta_values_permutation = delta_adata_permutation.X.mean(axis=0)

    # Define dynamic threshold and binarize
    avg_perturbation = np.mean(delta_values_permutation)
    delta_values_binary = np.where(delta_values > avg_perturbation, 1, 0).astype(np.int32)

    # Reconstruct bin coordinates from raw delta peaks
    coords = delta_adata.var_names.to_series().str.extract(r'(?P<chrom>chr[^_:-]+)[_:-](?P<start>\d+)[_:-](?P<end>\d+)')
    coords['start'] = coords['start'].astype(int)
    coords['bin_start'] = (coords['start'] // bin_size) * bin_size
    coords['bin_end'] = coords['bin_start'] + bin_size
    coords['new_region'] = coords['chrom'] + '-' + coords['bin_start'].astype(str) + '-' + coords['bin_end'].astype(str)

    # Max Pooling: Take the maximum perturbation score within each bin
    df = pd.DataFrame({
        'delta': delta_values,
        'delta_binary': delta_values_binary,
        'bin': coords['new_region'].values
    })
    df_max = df.groupby('bin').max()
    
    # Align aggregated scores to Ground Truth bin indices (missing bins filled with 0)
    delta_series = df_max['delta'].reindex(labels.index).fillna(0)
    delta_binary_series = df_max['delta_binary'].reindex(labels.index).fillna(0).astype(int)

    aupr_results = {}
    # Process metrics if labels exist for the target cell type
    if ct in labels.columns:
        ct_labels = labels[ct]
        
        for dis in distances:
            search_start = gene_start - dis
            search_end = gene_end + dis
            target_bins = [
                b for b in labels.index
                for p in [b.split('-')]
                if p[0] == gene_chrom and int(p[2]) >= search_start and int(p[1]) <= search_end
            ]
            
            # Align Ground Truth labels and pooled predictions
            aligned_data = pd.DataFrame({
                'label': ct_labels[target_bins],
                'delta': delta_series[target_bins],
                'delta_binary': delta_binary_series[target_bins],
            }).dropna()
            print(f"positive peak: {aligned_data['delta_binary'].sum()}")
            
            pos_ratio = aligned_data['label'].mean()
            if pos_ratio == 0 or pos_ratio == 1:
                print(f"Warning: All labels are same for cell type {ct}, AUPR undefined")
                aupr, auroc, mcc, f1 = np.nan, np.nan, np.nan, np.nan
            else:
                try:
                    aupr = average_precision_score(aligned_data['label'], aligned_data['delta'])
                    auroc = roc_auc_score(aligned_data['label'], aligned_data['delta'])
                    mcc = matthews_corrcoef(aligned_data['label'], aligned_data['delta_binary'])
                    f1 = f1_score(aligned_data['label'], aligned_data['delta_binary'])
                except Exception as e:
                    print(f"Error calculating metrics for {ct}: {e}")
                    aupr, auroc, mcc, f1 = np.nan, np.nan, np.nan, np.nan
            
            aupr_results[str(dis)] = {
                'dis': str(dis),
                'gene_name': gene_name,
                'aupr_delta_vs_label': aupr,
                'auroc_delta_vs_label': auroc,
                'mcc_delta_vs_label': mcc,
                'f1_delta_vs_label': f1,
                'n_total_peaks': len(aligned_data),
                'n_positive_peaks': int(aligned_data['label'].sum()),
                'positive_ratio': pos_ratio,
                'mean_delta': aligned_data['delta'].mean(),
                'std_delta': aligned_data['delta'].std()
            }
            
    return pd.DataFrame(aupr_results).T


def merge_fragment(adata, bin_size=10000):
    """
    Merge raw ATAC peaks into larger fixed-size genomic bins (e.g., 10kb).
    """
    coords = adata.var_names.to_series().str.extract(r'(?P<chrom>chr[^_:-]+)[_:-](?P<start>\d+)[_:-](?P<end>\d+)')
    if coords.isna().any().any():
        print("warning: peaks merge failed")

    coords['start'] = coords['start'].astype(int)
    coords['bin_start'] = (coords['start'] // bin_size) * bin_size
    coords['bin_end'] = coords['bin_start'] + bin_size
    coords['new_region'] = coords['chrom'] + '-' + coords['bin_start'].astype(str) + '-' + coords['bin_end'].astype(str)

    new_regions = pd.Categorical(coords['new_region'])

    # Mapping matrix (shape: peak × bin)
    row_indices = np.arange(adata.n_vars) 
    col_indices = new_regions.codes     
    data = np.ones(adata.n_vars)

    mapping_matrix = sp.coo_matrix(
        (data, (row_indices, col_indices)), 
        shape=(adata.n_vars, len(new_regions.categories))
    ).tocsc()

    # Matrix multiplication: (n_cells, n_old_peaks) @ (n_old_peaks, n_new_bins) = (n_cells, n_new_bins)
    new_X = adata.X @ mapping_matrix
    new_X.data = np.ones_like(new_X.data, dtype=np.int8)  # Binarize accessible bins
    new_var = pd.DataFrame(index=new_regions.categories)
    adata_bin = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)

    print(f"pre-merge adata: {adata.shape}")
    print(f"post-merge adata: {adata_bin.shape}")
    return adata_bin


args = parse_cal_label_args()
tis = args.tis
ct = args.ct
data_dir = args.data_dir
bin_size = 10000

# 1. Setup Sample to Tissue Mapping
obs_tissue_dict = {'CE336E1-S1': 'uterus', 'CE354E1-S1': 'uterus', 'CE357E1-S1': 'uterus', 'HT235B1-S1H1': 'breast', 'HT243B1-S1H4': 'breast', 'HT263B1-S1H1': 'breast', 'HT297B1-S1H1': 'breast', 'HT305B1-S1H1': 'breast', 'HT497B1-S1H1': 'breast', 'HT514B1-S1H3': 'breast', 'HT545B1-S1H1': 'breast', 'CE346E1-S1': 'uterus', 'CE347E1-S1K1': 'uterus', 'CE348E1-S1K1': 'uterus', 'CE349E1-S1': 'uterus', 'CE352E1-S1K1': 'uterus', 'CE356E1-S1': 'uterus', 'CM268C1-S1': 'colon', 'CM268C1-T1': 'colon', 'HT230C1-TH1': 'colon', 'HT250C1-TH1K1': 'colon', 'HT253C1-TH1K1': 'colon', 'HT291C1-M1A3': 'colon', 'HT307C1-TH1K1': 'colon', 'SP369H1-MC1': 'colon', 'HTX003S1-S1A1': 'tongue', 'HTX012S1-S1': 'tongue', 'HTX013S1-S1': 'tongue', 'P5216-N1': 'tongue', 'P5216-N2': 'tongue', 'P5504-N1': 'tongue', 'P5532-N1': 'tongue', 'P5576-1N1': 'tongue', 'P5590-N1': 'tongue', 'HT090P1-T2A3': 'pancreas', 'HT113P1-T2A3': 'pancreas', 'HT181P1-T1A3': 'pancreas', 'HT224P1-S1': 'pancreas', 'HT231P1-S1H3': 'pancreas', 'HT232P1-S1H1': 'pancreas', 'HT242P1-S1H1': 'pancreas', 'HT259P1-S1H1': 'pancreas', 'HT264P1-S1H2': 'pancreas', 'HT270P1-S1H2': 'pancreas', 'HT270P2-TH1FC1': 'pancreas', 'HT288P1-S1H4': 'pancreas', 'HT306P1-S1H1': 'pancreas', 'HT341P1-S1H1': 'pancreas', 'HT390P1-S1H1': 'pancreas', 'HT412P1-S1H1': 'pancreas', 'HT447P1-TH1K1A3': 'pancreas', 'HT452P1-TH1K1': 'pancreas', 'ML1199M1-TY1': 'skin', 'ML1199M1-TY2': 'skin', 'ML1232M1-TY1': 'skin', 'ML1239M1-TF1': 'skin', 'ML1294M1-TY1': 'skin', 'ML1332M1-TY1': 'skin', 'ML1511M1-S1': 'skin', 'ML1525M1-TY1': 'skin', 'ML1526M1-TA1': 'skin', 'CPT1541DU-S1': 'uterus', 'CPT1541DU-T1': 'uterus', 'CPT2373DU-S1': 'uterus', 'CPT2373DU-T1': 'uterus', 'CPT704DU-M1': 'uterus', 'CPT704DU-S1': 'uterus', 'CPT704DU-T1': 'uterus'}

# 2. Load and Preprocess Datasets
rna_list = ad.read_h5ad(data_dir + "/tumor/rna_tumor.h5ad")
atac_list = ad.read_h5ad(data_dir + "/tumor/tumor_atac.h5ad")

# Standardize Metadata
rna_list.obs['tissue'] = [obs_tissue_dict[x] for x in rna_list.obs.batch.values]
atac_list.obs['tissue'] = [obs_tissue_dict[x] for x in atac_list.obs.batch.values]
celltype_key='type'
rna_list.obs[celltype_key] = np.where(rna_list.obs.cell_type == 'Tumor', 'Tumor', 'Normal')
atac_list.obs[celltype_key] = np.where(atac_list.obs.cell_type == 'Tumor', 'Tumor', 'Normal')

# Configuration Variables
celltypes = ['Tumor', 'Normal']
distances=[5e5, 4e5, 3e5, 2e5, 1e5]
L = 500000
top = 10
use_absolute=True
pval_threshold=0.05
corr_threshold=0.01
min_pct=0.05

# Normalize RNA globally
rna_list_mat = rna_list.copy()
sc.pp.normalize_total(rna_list_mat, 1e4)
sc.pp.log1p(rna_list_mat)

# Bin merging and TF-IDF applied to ATAC globally
atac_list_bin = merge_fragment(atac_list, bin_size)
atac_list_bin.X = tfidf_seurat(atac_list_bin)

# Load annotations
gene_range_df = get_genes_from_gencode(os.path.join(data_dir, "gencode.v47.annotation.gtf"))
aupr_result = pd.DataFrame()
gene_pd = pd.read_csv(f"{data_dir}/perturbation_tumor/diff_exp_tumor_normal_gene_names_{tis}.csv")

# 3. Ensure Output Directories Exist
out_dir = os.path.join(data_dir, "perturbation_tumor", "gene_peak_nearby_merge")
os.makedirs(out_dir, exist_ok=True)  # Safety protection for file saving

# 4. Main Evaluation Loop: Target Genes
for gene_name in gene_pd[ct].dropna().values[:top]:
    
    # Determine genomic locus of target gene
    gene_info = gene_range_df[gene_range_df['name'] == gene_name]
    gene_chrom = gene_info['chrom'].values[0]
    gene_s = gene_info['start'].values[0]
    gene_e = gene_info['end'].values[0]
    gene_start = gene_s if gene_s < gene_e else gene_e
    gene_end = gene_e if gene_s < gene_e else gene_s
    search_start, search_end = gene_start - L, gene_end + L
    print(f"Analyzing tissue: {tis}, gene: {gene_name}, chrom: {gene_chrom}, gene_start: {gene_start}, gene_end: {gene_end}")

    # Load Perturbation Results (Actual vs Permutation)
    gene_atac_delta_adata = ad.read_h5ad(data_dir + f"/perturbation_tumor/gene_peak_regulation/atac_delta_tumor_normal_{ct}_{gene_name}_{gene_chrom}.h5ad")
    gene_atac_delta_adata_permutation = ad.read_h5ad(data_dir + f"/perturbation_tumor/gene_peak_regulation_permutation/atac_delta_tumor_normal_{ct}_{gene_name}_{gene_chrom}_permutation.h5ad")
    
    # Sync metadata for loaded perturbations
    gene_atac_delta_adata.obs['tissue'] = [obs_tissue_dict[x] for x in gene_atac_delta_adata.obs.batch.values]
    gene_atac_delta_adata_permutation.obs['tissue'] = [obs_tissue_dict[x] for x in gene_atac_delta_adata_permutation.obs.batch.values]
    gene_atac_delta_adata.obs[celltype_key] = np.where(gene_atac_delta_adata.obs.cell_type == 'Tumor', 'Tumor', 'Normal')
    gene_atac_delta_adata_permutation.obs[celltype_key] = np.where(gene_atac_delta_adata_permutation.obs.cell_type == 'Tumor', 'Tumor', 'Normal')

    # Filter out irrelevant ATAC bins far from the gene body (Replaced Lambda with list comprehension)
    peaks = [
        peak for peak in atac_list_bin.var_names 
        if (lambda p: (p[0]==gene_chrom and int(p[2])>=search_start and int(p[1])<=search_end))
        (peak.split('-'))
    ]
    atac_test_mat = atac_list_bin[rna_list_mat.obs_names, peaks].copy()
    
    # Filter Perturbation matrices (raw peaks) to only contain targeted cell types and tissues
    mask_delta = (gene_atac_delta_adata.obs.tissue == tis) & (gene_atac_delta_adata.obs[celltype_key] == ct)
    gene_atac_delta_adata = gene_atac_delta_adata[mask_delta]
    
    mask_perm = (gene_atac_delta_adata_permutation.obs.tissue == tis) & (gene_atac_delta_adata_permutation.obs[celltype_key] == ct)
    gene_atac_delta_adata_permutation = gene_atac_delta_adata_permutation[mask_perm]
    
    # Prepare biological evaluation matrices
    rna_test_mat = rna_list_mat[rna_list_mat.obs.tissue==tis]
    atac_test_mat = atac_test_mat[atac_test_mat.obs.tissue==tis]

    print("=" * 60)
    print("rna_test_mat shape: ", rna_test_mat.shape)
    print("=" * 60)

    # 5. Generate or Load Ground Truth Labels via correlation Analysis
    label_path = os.path.join(out_dir, f"{tis}_{gene_name}_{ct}_nearby_labels_{bin_size}.csv")
    if not os.path.exists(label_path):
        correlations, p_values, adj_p_values, labels = \
            calculate_celltype_specific_correlation_with_labels(
                rna_test_mat, atac_test_mat, gene_name, celltype_key, ct,
                use_absolute, min_pct=min_pct, pval_threshold=pval_threshold,
                corr_threshold=corr_threshold
            )
        labels.to_csv(label_path)
        del correlations, p_values, adj_p_values
    else:
        labels = pd.read_csv(label_path, index_col=0)

    # 6. Evaluate Perturbation Model using Ground Truth Labels
    aupr_delta_labels = calculate_delta_aupr_for_celltype(
        gene_atac_delta_adata, gene_atac_delta_adata_permutation, labels, gene_name, gene_chrom, gene_start, gene_end, ct, distances, bin_size, use_absolute
    )
    aupr_delta_labels['gene_chrom'] = [gene_chrom] * len(distances)
    aupr_delta_labels['gene_start'] = [gene_start] * len(distances)
    aupr_delta_labels['gene_end'] = [gene_end] * len(distances)

    del gene_atac_delta_adata, gene_atac_delta_adata_permutation, rna_test_mat, atac_test_mat

    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(aupr_delta_labels[['aupr_delta_vs_label', 'auroc_delta_vs_label', 'mcc_delta_vs_label', 'f1_delta_vs_label', 'positive_ratio', 'n_total_peaks', 'n_positive_peaks']].round(4))
    
    # Accumulate results for the batch
    aupr_result = pd.concat([aupr_result, aupr_delta_labels], ignore_index=True)

# 7. Save final benchmark results
aupr_result.to_csv(os.path.join(out_dir, f"{tis}_nearby_aupr_{ct}_permutation_{bin_size}.csv"))