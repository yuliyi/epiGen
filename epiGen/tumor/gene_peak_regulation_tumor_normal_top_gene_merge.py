import os
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score
from utils import parse_cal_label_args, tfidf_seurat, top_k_metrics, get_genes_from_gencode, calculate_celltype_specific_correlation_with_labels
import warnings
warnings.filterwarnings('ignore')


def calculate_delta_aupr_for_celltype(delta_adata, pred_correlations, labels, pred_labels, 
                                      gene_name, gene_chrom, gene_start, gene_end,
                                      ct='Tumor', distances=[5e5, 3e5, 1e5], bin_size=10000):
    """
    Evaluate the prediction performance (AUPR, AUROC, F1, MCC, and top@K) of perturbation delta and correlation-based scores
    against Ground Truth labels across different proximity windows.
    """
    
    # Aggregate perturbations across cells
    delta_values = np.abs(delta_adata.X).mean(axis=0)

    # Reconstruct bin coordinates from raw delta peaks
    coords = delta_adata.var_names.to_series().str.extract(r'(?P<chrom>chr[^_:-]+)[_:-](?P<start>\d+)[_:-](?P<end>\d+)')
    coords['start'] = coords['start'].astype(int)
    coords['end'] = coords['end'].astype(int)
    coords['bin_start'] = (coords['start'] // bin_size) * bin_size
    coords['bin_end'] = coords['bin_start'] + bin_size
    coords['new_region'] = coords['chrom'] + '-' + coords['bin_start'].astype(str) + '-' + coords['bin_end'].astype(str)
    # Store all peak-level information
    df_base = pd.DataFrame({
        'chrom': coords['chrom'].values,
        'start': coords['start'].values,
        'end': coords['end'].values,
        'delta': delta_values,
        'bin': coords['new_region'].values
    })

    aupr_results = {}
    # Process metrics if labels exist for the target cell type
    if ct in labels.columns:
        ct_labels = labels[ct]
        ct_preds_labels = pred_labels[ct]
        ct_preds_correlations = pred_correlations[ct]
        
        for dis in distances:
            search_start = gene_start - dis
            search_end = gene_end + dis
            # Filter raw peaks within the current local distance window
            mask_local = (df_base['chrom'] == gene_chrom) & \
                         (df_base['start'] <= search_end) & \
                         (df_base['end'] >= search_start)
            df_local = df_base[mask_local].copy()

            # Calculate local dynamic threshold and binarization
            X = df_local['delta'].values.reshape(-1, 1)
            try:
                best_gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=42).fit(X)
                means = best_gmm.means_.flatten()
                sorted_means = np.sort(means)
                mid_mean_threshold = sorted_means[1]
                df_local['delta_binary'] = (df_local['delta'] >= mid_mean_threshold).astype(int)
            except Exception as e:
                print(f"GMM fitting failed for distance {dis}: {e}. Fallback to percentile.")
                threshold = np.percentile(df_local['delta'], bin_thres)
                df_local['delta_binary'] = np.where(df_local['delta'] > threshold, 1, 0).astype(np.int32)

            # Perform Max Pooling based on current local results
            df_max = df_local[['bin', 'delta', 'delta_binary']].groupby('bin').max()

            target_bins = [
                b for b in labels.index
                for p in [b.split('-')]
                if p[0] == gene_chrom and int(p[2]) >= search_start and int(p[1]) <= search_end
            ]
            # Align aggregated scores to Ground Truth bin indices
            delta_series = df_max['delta'].reindex(target_bins).fillna(0)
            delta_binary_series = df_max['delta_binary'].reindex(target_bins).fillna(0).astype(int)
            
            # Align Ground Truth labels and pooled predictions
            aligned_data = pd.DataFrame({
                'label': ct_labels[target_bins],
                'delta': delta_series,
                'delta_binary': delta_binary_series,
                'preds': ct_preds_correlations[target_bins],
                'preds_label': ct_preds_labels[target_bins],
            }).dropna()
            print(f"positive peak: {aligned_data['delta_binary'].sum()}")
            
            pos_ratio = aligned_data['label'].mean()
            if pos_ratio == 0 or pos_ratio == 1:
                print(f"Warning: All labels are same for cell type {ct}, AUPR undefined")
                aupr = np.nan
                auroc = np.nan
                mcc = np.nan
                f1 = np.nan
                prec10_delta = np.nan
                prec50_delta = np.nan
                prec100_delta = np.nan
                rec10_delta = np.nan
                rec50_delta = np.nan
                rec100_delta = np.nan
                preds_aupr = np.nan
                preds_auroc = np.nan
                preds_mcc = np.nan
                preds_f1 = np.nan
                prec10_preds = np.nan
                prec50_preds = np.nan
                prec100_preds = np.nan
                rec10_preds = np.nan
                rec50_preds = np.nan
                rec100_preds = np.nan
            else:
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
                    prec10_delta, rec10_delta = top_k_metrics(aligned_data['label'].values, aligned_data['delta'].values, 10)
                    prec50_delta, rec50_delta = top_k_metrics(aligned_data['label'].values, aligned_data['delta'].values, 50)
                    prec100_delta, rec100_delta = top_k_metrics(aligned_data['label'].values, aligned_data['delta'].values, 100)
                except Exception as e:
                    print(f"Error calculating metrics for {ct}: {e}")
                    aupr = np.nan
                    auroc = np.nan
                    mcc = np.nan
                    f1 = np.nan
                    prec10_delta = np.nan
                    prec50_delta = np.nan
                    prec100_delta = np.nan
                    rec10_delta = np.nan
                    rec50_delta = np.nan
                    rec100_delta = np.nan

                try:
                    preds_aupr = average_precision_score(
                        aligned_data['label'],
                        aligned_data['preds']
                    )
                    preds_auroc = roc_auc_score(
                        aligned_data['label'],
                        aligned_data['preds']
                    )
                    preds_mcc = matthews_corrcoef(aligned_data['label'], aligned_data['preds_label'])
                    preds_f1 = f1_score(aligned_data['label'], aligned_data['preds_label'])
                    prec10_preds, rec10_preds = top_k_metrics(aligned_data['label'].values, aligned_data['preds'].values, 10)
                    prec50_preds, rec50_preds = top_k_metrics(aligned_data['label'].values, aligned_data['preds'].values, 50)
                    prec100_preds, rec100_preds = top_k_metrics(aligned_data['label'].values, aligned_data['preds'].values, 100)
                except Exception as e:
                    print(f"Error calculating metrics for {ct}: {e}")
                    preds_aupr = np.nan
                    preds_auroc = np.nan
                    preds_mcc = np.nan
                    preds_f1 = np.nan
                    prec10_preds = np.nan
                    prec50_preds = np.nan
                    prec100_preds = np.nan
                    rec10_preds = np.nan
                    rec50_preds = np.nan
                    rec100_preds = np.nan
            
            aupr_results[str(dis)] = {
                'dis': str(dis),
                'gene_name': gene_name,
                'aupr_delta_vs_label': aupr,
                'auroc_delta_vs_label': auroc,
                'mcc_delta_vs_label': mcc,
                'f1_delta_vs_label': f1,
                'top10_prec_delta_vs_label': prec10_delta,
                'top50_prec_delta_vs_label': prec50_delta,
                'top100_prec_delta_vs_label': prec100_delta,
                'top10_rec_delta_vs_label': rec10_delta,
                'top50_rec_delta_vs_label': rec50_delta,
                'top100_rec_delta_vs_label': rec100_delta,
                'auroc_preds_vs_label': preds_auroc,
                'aupr_preds_vs_label': preds_aupr,
                'mcc_preds_vs_label': preds_mcc,
                'f1_preds_vs_label': preds_f1,
                'top10_prec_preds_vs_label': prec10_preds,
                'top50_prec_preds_vs_label': prec50_preds,
                'top100_prec_preds_vs_label': prec100_preds,
                'top10_rec_preds_vs_label': rec10_preds,
                'top50_rec_preds_vs_label': rec50_preds,
                'top100_rec_preds_vs_label': rec100_preds,
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
pval_threshold=0.05
corr_threshold=0.01
min_pct=0.05
bin_thres = 98
# Normalize RNA globally
rna_list_mat = rna_list.copy()
sc.pp.normalize_total(rna_list_mat, 1e4)
sc.pp.log1p(rna_list_mat)

# Bin merging and TF-IDF applied to ATAC globally
atac_list_bin = merge_fragment(atac_list, bin_size)
atac_list_bin.X = tfidf_seurat(atac_list_bin)
atac_tumor_pred = ad.read_h5ad(data_dir + "/prediction/tumor/allchr_tumor_tumorcell_bins.h5ad")
atac_normal_pred = ad.read_h5ad(data_dir + "/prediction/tumor/allchr_tumor_normalcell_bins.h5ad")
atac_normal_pred.X = atac_normal_pred.X.astype('int8')
atac_tumor_pred.obs[celltype_key] = 'Tumor'
atac_normal_pred.obs[celltype_key] = 'Normal'
atac_pred = ad.concat([atac_tumor_pred, atac_normal_pred], axis=0)
atac_pred.obs['tissue'] = [obs_tissue_dict[x] for x in atac_pred.obs.batch.values]
atac_pred.X = tfidf_seurat(atac_pred)
atac_pred_bin = merge_fragment(atac_pred, bin_size)
atac_pred_bin.X = tfidf_seurat(atac_pred_bin)

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
    gene_start = min(gene_s, gene_e)
    gene_end = max(gene_s, gene_e)
    search_start, search_end = gene_start - L, gene_end + L
    print(f"Analyzing tissue: {tis}, gene: {gene_name}, chrom: {gene_chrom}, gene_start: {gene_start}, gene_end: {gene_end}")

    # Load Perturbation Results
    gene_atac_delta_adata = ad.read_h5ad(data_dir + f"/perturbation_tumor/gene_peak_regulation/atac_delta_tumor_normal_{ct}_{gene_name}_{gene_chrom}.h5ad")
    
    # Sync metadata for loaded perturbations
    gene_atac_delta_adata.obs['tissue'] = [obs_tissue_dict[x] for x in gene_atac_delta_adata.obs.batch.values]
    gene_atac_delta_adata.obs[celltype_key] = np.where(gene_atac_delta_adata.obs.cell_type == 'Tumor', 'Tumor', 'Normal')

    # Filter out irrelevant ATAC bins far from the gene body (Replaced Lambda with list comprehension)
    peaks = [
        peak for peak in atac_list_bin.var_names 
        if (lambda p: (p[0]==gene_chrom and int(p[2])>=search_start and int(p[1])<=search_end))
        (peak.split('-'))
    ]
    atac_test_mat = atac_list_bin[rna_list_mat.obs_names, peaks].copy()
    atac_pred_mat = atac_pred_bin[rna_list_mat.obs_names, peaks].copy()
    
    # Filter Perturbation matrices (raw peaks) to only contain targeted cell types and tissues
    mask_delta = (gene_atac_delta_adata.obs.tissue == tis) & (gene_atac_delta_adata.obs[celltype_key] == ct)
    gene_atac_delta_adata = gene_atac_delta_adata[mask_delta]
    
    # Prepare biological evaluation matrices
    rna_test_mat = rna_list_mat[rna_list_mat.obs.tissue==tis]
    atac_test_mat = atac_test_mat[atac_test_mat.obs.tissue==tis]
    atac_pred_mat = atac_pred_mat[atac_pred_mat.obs.tissue==tis]

    print("=" * 60)
    print("rna_test_mat shape: ", rna_test_mat.shape)
    print("=" * 60)

    # 5. Generate or Load Ground Truth Labels via correlation Analysis
    label_path = os.path.join(out_dir, f"{tis}_{gene_name}_{ct}_nearby_labels_{bin_size}.csv")
    if not os.path.exists(label_path):
        correlations, p_values, adj_p_values, labels = \
            calculate_celltype_specific_correlation_with_labels(
                rna_test_mat, atac_test_mat, gene_name, celltype_key, ct,
                min_pct=min_pct, pval_threshold=pval_threshold,
                corr_threshold=corr_threshold
            )
        labels.to_csv(label_path)
        del correlations, p_values, adj_p_values
    else:
        labels = pd.read_csv(label_path, index_col=0)

    pred_label_path = os.path.join(out_dir, f"{tis}_{gene_name}_{ct}_nearby_pred_labels_{bin_size}.csv")
    if not os.path.exists(pred_label_path):
        pred_correlations, _, _, pred_labels = \
            calculate_celltype_specific_correlation_with_labels(
                rna_test_mat, atac_pred_mat, gene_name, celltype_key, ct,
                min_pct=min_pct, pval_threshold=pval_threshold,
                corr_threshold=corr_threshold
            )
        pred_correlations.to_csv(out_dir + f"/{tis}_{gene_name}_{ct}_nearby_pred_correlations_{bin_size}.csv")
        pred_labels.to_csv(pred_label_path)
    else:
        pred_correlations = pd.read_csv(out_dir + f"/{tis}_{gene_name}_{ct}_nearby_pred_correlations_{bin_size}.csv", index_col=0)
        pred_labels = pd.read_csv(out_dir, index_col=0)

    # 6. Evaluate Perturbation Model using Ground Truth Labels
    aupr_delta_labels = calculate_delta_aupr_for_celltype(
        gene_atac_delta_adata, pred_correlations, labels, pred_labels, gene_name, gene_chrom, gene_start, gene_end, ct, distances, bin_size
    )
    aupr_delta_labels['gene_chrom'] = [gene_chrom] * len(distances)
    aupr_delta_labels['gene_start'] = [gene_start] * len(distances)
    aupr_delta_labels['gene_end'] = [gene_end] * len(distances)

    del gene_atac_delta_adata, pred_correlations, labels, pred_labels, rna_test_mat, atac_test_mat, atac_pred_mat

    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(aupr_delta_labels[['aupr_delta_vs_label', 'auroc_delta_vs_label', 'mcc_delta_vs_label', 'f1_delta_vs_label', 'positive_ratio', 'n_total_peaks', 'n_positive_peaks']].round(4))
    
    # Accumulate results for the batch
    aupr_result = pd.concat([aupr_result, aupr_delta_labels], ignore_index=True)

# 7. Save final benchmark results
aupr_result.to_csv(os.path.join(out_dir, f"{tis}_nearby_aupr_{ct}_permutation_{bin_size}.csv"))