import argparse
import os
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from pyranges import read_gtf
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments for hyperparameters and paths."""
    parser = argparse.ArgumentParser(description="Run epiGen.")
    parser.add_argument('--data_dir', default='/path/data',
                        help='Input data path.')
    parser.add_argument('--ct', type=str, default='T', help='cell type (e.g. T/B/NK)')
    args = parser.parse_args()
    return args


def tfidf_seurat(adata: ad.AnnData, scale_factor=10000) -> None:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
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
    
    # 3. TF-IDF
    if scipy.sparse.issparse(tf):
        X_tfidf = tf.multiply(idf)
        X_tfidf.data = np.log1p(X_tfidf.data)  # np.log1p = log(1+x)
        if hasattr(X_tfidf, 'tocsr'):
            X_tfidf = X_tfidf.tocsr()
    else:
        X_tfidf = tf * idf
        X_tfidf = np.log1p(X_tfidf)
    return X_tfidf


def calculate_celltype_specific_correlation_with_labels(rna_adata, atac_adata, gene_name='LEF1', 
                                                       celltype_key='cell_type', ct='T',
                                                       min_pct=0.05, pval_threshold=0.05, corr_threshold=0.1):
    """
    Calculate the Pearson correlation between a specific gene and all ATAC peaks 
    within a target cell type, generating binary labels based on significance thresholds.
    """
    common_cells = rna_adata.obs_names
    
    if gene_name not in rna_adata.var.gene_name.values:
        raise ValueError(f"Gene {gene_name} not found in RNA data")
    
    # Extract gene expression array
    gene_expr = pd.Series(
        rna_adata[common_cells, rna_adata.var.gene_name==gene_name].X.toarray().flatten(),
        index=common_cells
    )
    
    # Extract ATAC data
    atac_data = pd.DataFrame(
        atac_adata[common_cells].X.toarray() if hasattr(atac_adata.X, 'toarray') else atac_adata[common_cells].X,
        index=common_cells,
        columns=atac_adata.var_names
    )
    
    # Initialize result dataframes
    correlations = pd.DataFrame(index=atac_data.columns, columns=[ct])
    p_values = pd.DataFrame(index=atac_data.columns, columns=[ct])
    labels = pd.DataFrame(0, index=atac_data.columns, columns=[ct])
    
    # Filter cells belonging to the target cell type
    ct_cells = atac_adata.obs[atac_adata.obs[celltype_key] == ct].index
    ct_cells = [c for c in ct_cells if c in common_cells]
    print(f"cell type: {ct}, cell num: {len(ct_cells)}")
    
    # Iterate through all peaks to calculate correlation with the target gene
    for peak in atac_data.columns:
        peak_data = atac_data.loc[:, peak]
        gene_data = gene_expr

        # Filter out peaks expressed in less than 'min_pct' of cells
        pct_expressed = (peak_data[ct_cells] > 0).sum() / len(ct_cells)
        if pct_expressed < min_pct:
            correlations.loc[peak, ct] = 0
            p_values.loc[peak, ct] = np.nan
            continue
        
        # Calculate Pearson correlation if there is variation in both peak and gene expression
        if len(np.unique(peak_data)) > 1 and len(np.unique(gene_data)) > 1:
            corr, pval = pearsonr(peak_data, gene_data)
            corr = abs(corr)
            correlations.loc[peak, ct] = corr
            p_values.loc[peak, ct] = pval
        else:
            correlations.loc[peak, ct] = 0
            p_values.loc[peak, ct] = np.nan
    
    # Benjamini-Hochberg FDR
    adjusted_p_values = p_values.copy()
    valid_pvals = p_values[ct].dropna()
    if len(valid_pvals) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            valid_pvals, alpha=pval_threshold, method='fdr_bh'
        )
        adjusted_p_values.loc[valid_pvals.index, ct] = pvals_corrected
        
        # Assign positive label (1) if the peak passes both FDR and correlation thresholds
        for peak in valid_pvals.index:
            corr_val = correlations.loc[peak, ct]
            adj_pval = pvals_corrected[valid_pvals.index.get_loc(peak)]
            
            if adj_pval < pval_threshold and abs(corr_val) > corr_threshold:
                labels.loc[peak, ct] = 1
            else:
                labels.loc[peak, ct] = 0
                correlations.loc[peak, ct] = 0
    
    return correlations, p_values, adjusted_p_values, labels


def calculate_delta_aupr_for_celltype(delta_adata, delta_adata_permutation, pred_correlations, labels, gene_name, gene_chrom, gene_start, gene_end,
                                      ct='T', distances=[5e5, 3e5, 1e5]):
    """
    Calculate classification metrics AUPRC for the perturbation delta 
    against the ground-truth peak labels at multiple genomic distance intervals.
    """
    bool_peak = []
    # Identify boolean masks for peaks falling within defined distance windows around the gene
    for dis in distances:
        search_start = gene_start - dis
        search_end = gene_end + dis
        peaks = [
            (lambda p: (p[0]==gene_chrom and int(p[2])>=search_start and int(p[1])<=search_end))
            (peak.split('-'))
            for peak in delta_adata.var_names
        ]
        bool_peak.append(peaks)
        
    aupr_results = {}
    delta_values = np.abs(delta_adata.X).mean(axis=0)
    delta_values_permutation = np.abs(delta_adata_permutation.X).mean(axis=0)

    if ct in labels.columns:
        ct_labels = labels[ct]
        ct_preds = pred_correlations[ct]
        
        for idx, dis in enumerate(distances):
            # Subset peak statistics specifically for this genomic distance interval
            delta_values_range = delta_values[bool_peak[idx]]

            # Use permutation background to set binarization threshold
            avg_perturbation = np.mean(delta_values_permutation[bool_peak[idx]])
            delta_values_binary = np.where(delta_values_range > avg_perturbation, 1, 0).astype(np.int32)

            aligned_data = pd.DataFrame({
                'label': ct_labels[bool_peak[idx]].values,
                'delta': delta_values_range,
                'delta_binary': delta_values_binary,
                'preds': ct_preds[bool_peak[idx]].values
            }).dropna()
            
            pos_ratio = aligned_data['label'].mean()
            if pos_ratio == 0 or pos_ratio == 1:
                print(f"Warning: All labels are same for cell type {ct}, AUPR undefined")
                aupr = np.nan
                auroc = np.nan
                mcc = np.nan
                f1 = np.nan
                preds_aupr = np.nan
                preds_auroc = np.nan
            else:
                # Calculate evaluation metrics comparing real label vs perturbation delta
                try:
                    aupr = average_precision_score(
                        aligned_data['label'],
                        aligned_data['delta']
                    )
                    auroc = roc_auc_score(
                        aligned_data['label'],
                        aligned_data['delta']
                    )
                    mcc = matthews_corrcoef(aligned_data['label'], aligned_data['delta_binary'])
                    f1 = f1_score(aligned_data['label'], aligned_data['delta_binary'])
                except Exception as e:
                    print(f"Error calculating metrics for {ct}: {e}")
                    aupr = np.nan
                    auroc = np.nan
                    mcc = np.nan
                    f1 = np.nan
                # Calculate metrics comparing real label vs raw model predicted correlation
                try:
                    preds_aupr = average_precision_score(
                        aligned_data['label'],
                        aligned_data['preds']
                    )
                    preds_auroc = roc_auc_score(
                        aligned_data['label'],
                        aligned_data['preds']
                    )
                except Exception as e:
                    print(f"Error calculating metrics for {ct}: {e}")
                    preds_aupr = np.nan
                    preds_auroc = np.nan
            
            aupr_results[str(dis)] = {
                'dis': str(dis),
                'gene_name': gene_name,
                'aupr_delta_vs_label': aupr,
                'auroc_delta_vs_label': auroc,
                'mcc_delta_vs_label': mcc,
                'f1_delta_vs_label': f1,
                'auroc_preds_vs_label': preds_auroc,
                'aupr_preds_vs_label': preds_aupr,
                'n_total_peaks': len(aligned_data),
                'n_positive_peaks': int(aligned_data['label'].sum()),
                'positive_ratio': pos_ratio,
                'mean_delta': aligned_data['delta'].mean(),
                'std_delta': aligned_data['delta'].std()
            }
    
    return pd.DataFrame(aupr_results).T


def get_genes_from_gencode(file):
    '''
    Extract gene name, ID, and chromosome coordinates for protein-coding 
    genes from a standard GENCODE GTF file.
    '''
    gtf = read_gtf(file).as_df()

    gene_list = gtf[(gtf.Feature == 'gene') & (
        gtf.gene_type == 'protein_coding')][['Chromosome', 'Start', 'End', 'gene_name', 'gene_id']]
    gene_list.columns = ['chrom', 'start', 'end', 'name', 'id']
    return (gene_list.convert_dtypes())


args = parse_args()
ct = args.ct
data_dir = args.data_dir
# Load multi-omics data and metadata
rna_list = ad.read_h5ad(data_dir + "/rna_data/rna_independence.h5ad")
atac_list = ad.read_h5ad(data_dir + f"/atac_data/atac_independence.h5ad")
celltype = pd.read_csv(os.path.join(data_dir + "/statistic/celltype.csv"), index_col=0)
rna_list.obs = celltype
atac_list.obs = celltype
# Define parameters for the perturbation and evaluation
celltypes = ['T', 'B', 'NK']
celltype_key='ct_a'
distances=[5e5, 4e5, 3e5, 2e5, 1e5]# Scanning windows (500kb to 100kb)
L = 500000
top = 20 # Evaluate top 20 genes
pval_threshold=0.05  # adjust p value
corr_threshold=0.01  # min correlation threshold
min_pct=0.05  # min percent of peaks expressed cell
rna_list_mat = rna_list[rna_list.obs.celltype != 'Unknown'].copy()
atac_list = atac_list[atac_list.obs.celltype != 'Unknown'].copy()
sc.pp.normalize_total(rna_list_mat, 1e4)
sc.pp.log1p(rna_list_mat)
atac_list.X = tfidf_seurat(atac_list)
# Load gene genomic references
gene_range_df = get_genes_from_gencode(os.path.join(data_dir, "gencode.v47.annotation.gtf.gz"))
aupr_result = pd.DataFrame()
# Load specific differentially expressed genes to simulate perturbation
gene_pd = pd.read_csv(f"{data_dir}/perturbation/diff_exp_gene_names.csv")
for gene_name in gene_pd[ct].dropna().values[:top]:
    # Locate gene position on the chromosome
    gene_info = gene_range_df[gene_range_df['name'] == gene_name]
    gene_chrom = gene_info['chrom'].values[0]
    gene_s = gene_info['start'].values[0]
    gene_e = gene_info['end'].values[0]
    gene_start = min(gene_s, gene_e)
    gene_end = max(gene_s, gene_e)
    search_start, search_end = gene_start - L, gene_end + L
    print(f"Analyzing gene: {gene_name}, chrom: {gene_chrom}, gene_start: {gene_start}, gene_end: {gene_end}")
    # Load calculated perturbation delta and randomized permutations for the given gene
    gene_atac_delta_adata = ad.read_h5ad(data_dir + f"/perturbation/gene_peak_regulation/atac_delta_{ct}_{gene_name}_{gene_chrom}.h5ad")
    gene_atac_delta_adata_permutation = ad.read_h5ad(data_dir + f"/perturbation/gene_peak_regulation_permutation/atac_delta_{ct}_{gene_name}_{gene_chrom}_permutation.h5ad")
    # Retrieve all peaks located within the maximum defined search window
    peaks = [
        peak for peak in atac_list.var_names 
        if (lambda p: (p[0]==gene_chrom and int(p[2])>=search_start and int(p[1])<=search_end))
        (peak.split('-'))
    ]
    # Subset matching genomic data and predictions
    atac_test_mat = atac_list[:, peaks].copy()
    pred_list = np.load(f"{data_dir}/prediction/{gene_chrom}_all_preds_nofinetune.npy")
    pred_list = ad.AnnData(pred_list)
    pred_list.obs = celltype
    pred_list.var = gene_atac_delta_adata.var.copy()
    # Restrict to celltypes of interest
    pred_test_mat = pred_list[pred_list.obs.ct_a.isin(celltypes), atac_test_mat.var_names]
    gene_atac_delta_adata = gene_atac_delta_adata[gene_atac_delta_adata.obs.ct_a==ct, atac_test_mat.var_names]
    gene_atac_delta_adata_permutation = gene_atac_delta_adata_permutation[gene_atac_delta_adata_permutation.obs.ct_a==ct, atac_test_mat.var_names]
    rna_test_mat = rna_list_mat[rna_list_mat.obs.ct_a.isin(celltypes)]
    atac_test_mat = atac_test_mat[atac_test_mat.obs.ct_a.isin(celltypes)]


    print("=" * 60)
    print("atac_test_mat shape: ", atac_test_mat.shape)
    print("=" * 60)
    # 1. Establish the "Ground Truth" correlation landscape between RNA expression and ATAC peak accessibility
    correlations, p_values, adj_p_values, labels = \
        calculate_celltype_specific_correlation_with_labels(
            rna_test_mat, atac_test_mat, gene_name, celltype_key, ct,
            min_pct=min_pct, pval_threshold=pval_threshold,
            corr_threshold=corr_threshold
        )
    labels.to_csv(data_dir + f"/perturbation/gene_peak_nearby/{gene_name}_nearby_labels.csv")
    # 2. Extract correlation matrix based on between RNA expression and predicted peak accessibility
    pred_correlations, _, _, _ = \
        calculate_celltype_specific_correlation_with_labels(
            rna_test_mat, pred_test_mat, gene_name, celltype_key, ct,
            min_pct=0, pval_threshold=pval_threshold,
            corr_threshold=corr_threshold
        )
    pred_correlations.to_csv(data_dir + f"/perturbation/gene_peak_nearby/{gene_name}_nearby_pred_correlations.csv")
    # 3. Evaluate performance of perturbation vs ground truths
    aupr_delta_labels = calculate_delta_aupr_for_celltype(
        gene_atac_delta_adata, gene_atac_delta_adata_permutation, pred_correlations, labels, gene_name, gene_chrom, gene_start, gene_end, ct, distances
    )
    aupr_delta_labels['gene_chrom'] = [gene_chrom] * len(distances)
    aupr_delta_labels['gene_start'] = [gene_start] * len(distances)
    aupr_delta_labels['gene_end'] = [gene_end] * len(distances)

    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(aupr_delta_labels[['aupr_delta_vs_label', 'auroc_delta_vs_label', 'mcc_delta_vs_label', 'f1_delta_vs_label', 'auroc_preds_vs_label', 'aupr_preds_vs_label', 'positive_ratio', 'n_total_peaks', 'n_positive_peaks']].round(4))
    
    aupr_result = pd.concat([aupr_result, aupr_delta_labels], ignore_index=True)

aupr_result.to_csv(data_dir + f"/perturbation/gene_peak_nearby/nearby_aupr_{ct}_permutation.csv")