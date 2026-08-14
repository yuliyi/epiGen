import os
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score
from utils import parse_cal_label_args, tfidf_seurat, top_k_metrics, get_genes_from_gencode, calculate_celltype_specific_correlation_with_labels
import warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture


def filter_peak_chrom(delta_adata, pred_correlations, labels, pred_labels, gene_name, gene_chrom, gene_start, gene_end,
                      ct='RGCs', distances=[5e5, 4e5, 3e5, 2e5, 1e5]):
    """
    Filter peaks that fall within specific distance windows of the target gene,
    then extract aligned ground truth labels and perturbation scores using window-wise permutation mean.
    """
    aligned_data = {}
    
    bool_peak = []
    for dis in distances:
        search_start = gene_start - dis
        search_end = gene_end + dis
        peaks = [
            (lambda p: (p[0]==gene_chrom and int(p[2])>=search_start and int(p[1])<=search_end))
            (peak.split('-'))
            for peak in delta_adata.var_names
        ]
        bool_peak.append(np.array(peaks, dtype=bool))
        
    # Aggregate perturbations across cells
    delta_values = np.abs(delta_adata.X).mean(axis=0)

    # Process metrics if labels exist for the target cell type
    if ct in labels.columns:
        ct_labels = labels[ct]
        ct_pred_labels = pred_labels[ct]
        ct_pred_correlations = pred_correlations[ct]
        
        for idx, dis in enumerate(distances):
            # Extract subset based on distance window
            delta_values_range = delta_values[bool_peak[idx]]
            if len(delta_values_range) == 0:
                aligned_data[str(dis)] = pd.DataFrame()
                continue
            
            # Use GMM to set binarization threshold
            X = delta_values_range.reshape(-1, 1)
            try:
                best_gmm = GaussianMixture(n_components=3, covariance_type='tied', random_state=42).fit(X)
                means = best_gmm.means_.flatten()
                signal_class = np.argmax(means)
                gmm_labels = best_gmm.predict(X)
                delta_values_binary = (gmm_labels == signal_class).astype(int)
            except Exception as e:
                print(f"GMM fitting failed for distance {dis}: {e}. Fallback to percentile.")
                threshold = np.percentile(delta_values_range, bin_thres)
                delta_values_binary = np.where(delta_values_range > threshold, 1, 0).astype(np.int32)

            # Align Ground Truth labels and predictions
            aligned_data[str(dis)] = pd.DataFrame({
                'gene_name': [gene_name] * len(delta_values_range),
                'label': ct_labels[bool_peak[idx]].values,
                'delta': delta_values_range,
                'delta_binary': delta_values_binary,
                'pred_labels': ct_pred_labels[bool_peak[idx]].values,
                'preds': ct_pred_correlations[bool_peak[idx]].values,
            }).dropna()
            
    return aligned_data


def calculate_delta_aupr_for_single_gene(aligned_df, gene_name, target_ct='RGCs'):
    """
    Compute classification metrics (AUPRC, AUROC, MCC, F1) for a single gene's aligned dataframe.
    Returns a dictionary of metrics.
    """
    if aligned_df is None or len(aligned_df) == 0:
        return {
            'gene_name': gene_name,
            'aupr_delta_vs_label': np.nan,
            'auroc_delta_vs_label': np.nan,
            'mcc_delta_vs_label': np.nan,
            'f1_delta_vs_label': np.nan,
            'top10_prec_delta_vs_label': np.nan,
            'top50_prec_delta_vs_label': np.nan,
            'top100_prec_delta_vs_label': np.nan,
            'top10_rec_delta_vs_label': np.nan,
            'top50_rec_delta_vs_label': np.nan,
            'top100_rec_delta_vs_label': np.nan,
            'aupr_pred_vs_label': np.nan,
            'auroc_pred_vs_label': np.nan,
            'mcc_pred_vs_label': np.nan,
            'f1_pred_vs_label': np.nan,
            'top10_prec_pred_vs_label': np.nan,
            'top50_prec_pred_vs_label': np.nan,
            'top100_prec_pred_vs_label': np.nan,
            'top10_rec_pred_vs_label': np.nan,
            'top50_rec_pred_vs_label': np.nan,
            'top100_rec_pred_vs_label': np.nan,
            'n_total_peaks': 0,
            'n_positive_peaks': 0,
            'positive_ratio': np.nan,
            'mean_delta': np.nan,
            'std_delta': np.nan
        }
        
    pos_ratio = aligned_df['label'].mean()
    
    # Prevent undefined metrics when class variance is zero (all 0s or all 1s)
    if pos_ratio == 0 or pos_ratio == 1:
        aupr, auroc, mcc, f1, prec10_delta, rec10_delta, prec50_delta, rec50_delta, prec100_delta, rec100_delta = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        preds_aupr, preds_auroc, preds_mcc, preds_f1, prec10_preds, rec10_preds, prec50_preds, rec50_preds, prec100_preds, rec100_preds = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        try:
            aupr = average_precision_score(aligned_df['label'], aligned_df['delta'])
            auroc = roc_auc_score(aligned_df['label'], aligned_df['delta'])
            mcc = matthews_corrcoef(aligned_df['label'], aligned_df['delta_binary'])
            f1 = f1_score(aligned_df['label'], aligned_df['delta_binary'])
            prec10_delta, rec10_delta = top_k_metrics(aligned_df['label'].values, aligned_df['delta'].values, 10)
            prec50_delta, rec50_delta = top_k_metrics(aligned_df['label'].values, aligned_df['delta'].values, 50)
            prec100_delta, rec100_delta = top_k_metrics(aligned_df['label'].values, aligned_df['delta'].values, 100)
        except Exception as e:
            print(f"Error calculating metrics for {gene_name} in {target_ct}: {e}")
            aupr, auroc, mcc, f1, prec10_delta, rec10_delta, prec50_delta, rec50_delta, prec100_delta, rec100_delta = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        try:
            preds_aupr = average_precision_score(aligned_df['label'], aligned_df['preds'])
            preds_auroc = roc_auc_score(aligned_df['label'], aligned_df['preds'])
            preds_mcc = matthews_corrcoef(aligned_df['label'], aligned_df['pred_labels'])
            preds_f1 = f1_score(aligned_df['label'], aligned_df['pred_labels'])
            prec10_preds, rec10_preds = top_k_metrics(aligned_df['label'].values, aligned_df['preds'].values, 10)
            prec50_preds, rec50_preds = top_k_metrics(aligned_df['label'].values, aligned_df['preds'].values, 50)
            prec100_preds, rec100_preds = top_k_metrics(aligned_df['label'].values, aligned_df['preds'].values, 100)
        except Exception as e:
            print(f"Error calculating metrics for {gene_name} in {target_ct}: {e}")
            preds_aupr, preds_auroc, preds_mcc, preds_f1, prec10_preds, rec10_preds, prec50_preds, rec50_preds, prec100_preds, rec100_preds = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
    return {
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
        'n_total_peaks': len(aligned_df),
        'n_positive_peaks': int(aligned_df['label'].sum()),
        'positive_ratio': pos_ratio,
        'mean_delta': aligned_df['delta'].mean(),
        'std_delta': aligned_df['delta'].std()
    }


# ==========================================
# Runtime Script Processing Block
# ==========================================
args = parse_cal_label_args()
ct = args.ct
data_dir = args.data_dir

# 2. Load and Preprocess Datasets
rna_list = ad.read_h5ad(os.path.join(data_dir, "rna_data/Multi_Fetal_19W4d_FR_rna_ct.h5ad"))
atac_list = ad.read_h5ad(os.path.join(data_dir, "atac_data/Multi_Fetal_19W4d_FR_atac_cpeak.h5ad"))
ct_map = {'RGCs': ['midget ganglion cell of retina', 'retinal ganglion cell', 'ON parasol ganglion cell', 'OFF parasol ganglion cell'],
          'RPCs': ['retinal progenitor cell'],
          'MCs': ['Mueller cell'],
          'ACs': ['GABAergic amacrine cell', 'amacrine cell', 'glycinergic amacrine cell', 'starburst amacrine cell'],
          'BCs': ['retinal bipolar neuron', 'diffuse bipolar 2 cell', 'diffuse bipolar 3b cell', 'diffuse bipolar 1 cell', 'diffuse bipolar 3a cell', 'rod bipolar cell', 'ON-bipolar cell', 'diffuse bipolar 6 cell', 'diffuse bipolar 4 cell'],
          'PCs': ['retinal rod cell', 'retinal cone cell', 'S cone cell']}
# Configuration Variables
celltype_key = 'Cluster'
distances = [5e5, 4e5, 3e5, 2e5, 1e5]
L = 500000
top = 10
pval_threshold = 0.05
corr_threshold = 0.01
min_pct = 0.05
bin_thres = 98
# Filter out matched cell type
all_types = [item for sublist in ct_map.values() for item in sublist]
rna_list_mat = rna_list[rna_list.obs.cell_type.isin(all_types)].copy()

# Apply TF-IDF normalization to ATAC accessibility values
atac_list.X = tfidf_seurat(atac_list)

rna_st = ad.read_h5ad(data_dir + f"/rna_data/rna_retina_st.h5ad")
atac_pred = ad.read_h5ad(os.path.join(data_dir, "prediction/retina/allchr_retina_bins.h5ad"))
atac_pred.obs = rna_st.obs.copy()
atac_pred.X = tfidf_seurat(atac_pred)
rna_st = rna_st[rna_st.obs.Cluster.isin(ct_map.keys())].copy()
atac_pred = atac_pred[atac_pred.obs.Cluster.isin(ct_map.keys())].copy()

# Load gene structural range annotations
gene_range_df = get_genes_from_gencode(os.path.join(data_dir, "gencode.v47.annotation.gtf"))
gene_pd = pd.read_csv(os.path.join(data_dir, "perturbation_retina/diff_exp_gene_names.csv"))

label_save_dir = os.path.join(data_dir, "perturbation_retina/gene_peak_nearby")
os.makedirs(label_save_dir, exist_ok=True) 

# Initialize a dictionary of lists to record metric results for each gene across different distances
# Key: distance (str), Value: list of dictionaries containing individual gene's metrics
distance_results_collector = {str(dis): [] for dis in distances}

# 3. Main Evaluation Loop: Iterating through Top 10 DEGs
for gene_name in gene_pd[ct].dropna().values[:top]:
    
    # Extract genomic locus coordinates of the current target gene
    gene_info = gene_range_df[gene_range_df['name'] == gene_name]
    if gene_info.empty:
        print(f"Skipping {gene_name}: not found in gene annotations.")
        continue
        
    gene_chrom = gene_info['chrom'].values[0]
    gene_s = gene_info['start'].values[0]
    gene_e = gene_info['end'].values[0]
    gene_start = min(gene_s, gene_e)
    gene_end = max(gene_s, gene_e)
    search_start, search_end = gene_start - L, gene_end + L
    print(f"Analyzing celltype: {ct}, gene: {gene_name}, chrom: {gene_chrom}, gene_start: {gene_start}, gene_end: {gene_end}")

    # Load file-based H5AD model outputs for actual and permuted conditions
    delta_path = os.path.join(data_dir, f"perturbation_retina/gene_peak_regulation/atac_delta_retina_{ct}_{gene_name}_{gene_chrom}.h5ad")
    
    if not os.path.exists(delta_path):
        print(f"Skipping {gene_name}: Perturbation output files missing.")
        continue
        
    gene_atac_delta_adata = ad.read_h5ad(delta_path)

    # Filter out irrelevant distal ATAC peaks to speed up matrix alignment
    peaks = [
        peak for peak in atac_list.var_names 
        if peak.split('-')[0] == gene_chrom and int(peak.split('-')[2]) >= search_start and int(peak.split('-')[1]) <= search_end
    ]
    atac_list_mat = atac_list[rna_list_mat.obs_names, peaks].copy()
    atac_pred_mat = atac_pred[:, peaks].copy()
    
    # Squeeze matrices to match cell-type specific conditions
    mask_delta = gene_atac_delta_adata.obs[celltype_key] == ct
    gene_atac_delta_adata = gene_atac_delta_adata[mask_delta, atac_list_mat.var_names]
    
    print("=" * 60)
    print("rna_test_mat shape: ", rna_list_mat.shape)
    print("=" * 60)
    
    # Construct Ground Truth Spearman correlation vectors
    label_path = os.path.join(label_save_dir, f"{gene_name}_{ct}_nearby_labels.csv")
    if not os.path.exists(label_path):
        correlations, p_values, adj_p_values, labels = \
            calculate_celltype_specific_correlation_with_labels(
                rna_list_mat, atac_list_mat, ct_map, gene_name, 'feature_name', 'cell_type', ct,
                min_pct=min_pct, pval_threshold=pval_threshold,
                corr_threshold=corr_threshold
            )
        labels.to_csv(label_path)
        del correlations, p_values, adj_p_values
    else:
        labels = pd.read_csv(label_path, index_col=0)

    pred_label_path = os.path.join(label_save_dir, f"{gene_name}_{ct}_nearby_pred_labels.csv")
    if not os.path.exists(pred_label_path):
        pred_correlations, _, _, pred_labels = \
            calculate_celltype_specific_correlation_with_labels(
                rna_st, atac_pred_mat, ct_map, gene_name, 'gene_name', 'Cluster', ct,
                min_pct=min_pct, pval_threshold=pval_threshold,
                corr_threshold=corr_threshold
            )
        pred_correlations.to_csv(label_save_dir + f"/{gene_name}_{ct}_nearby_pred_correlations.csv")
        pred_labels.to_csv(pred_label_path)
    else:
        pred_correlations = pd.read_csv(label_save_dir + f"/{gene_name}_{ct}_nearby_pred_correlations.csv", index_col=0)
        pred_labels = pd.read_csv(pred_label_path, index_col=0)

    # Evaluate model accuracy inside the specific target gene windows
    aligned_data = filter_peak_chrom(
        gene_atac_delta_adata, pred_correlations, labels, pred_labels, gene_name, gene_chrom, gene_start, gene_end, ct, distances
    )
    
    # Calculate and store individual metrics for each distance window of the current gene
    for dis in distances:
        dis_str = str(dis)
        current_df = aligned_data.get(dis_str, pd.DataFrame())
        metrics = calculate_delta_aupr_for_single_gene(current_df, gene_name, target_ct=ct)
        distance_results_collector[dis_str].append(metrics)

# ==========================================
# 4. Final Aggregated Metric Processing (Averaged Approach)
# ==========================================
all_distance_dfs = []

for dis in distance_results_collector.keys():
    gene_list_for_dis = distance_results_collector[dis]
    if len(gene_list_for_dis) == 0:
        print(f"Warning: No valid gene metrics calculated for distance {dis}")
        continue
        
    # Convert individual gene metrics for the current distance into a DataFrame
    df_dis_individual = pd.DataFrame(gene_list_for_dis)
    
    # Add a tracker column for the distance window
    df_dis_individual['distance'] = str(dis)
    df_dis_individual['cell_type'] = ct
    
    # Reorder columns to place metadata at the front for academic neatness
    cols = ['cell_type', 'distance', 'gene_name'] + \
           [c for c in df_dis_individual.columns if c not in ['cell_type', 'distance', 'gene_name']]
    df_dis_individual = df_dis_individual[cols]
    
    all_distance_dfs.append(df_dis_individual)

if all_distance_dfs:
    # Concatenate all rows across all genes and all distance windows
    full_metrics_df = pd.concat(all_distance_dfs, axis=0).reset_index(drop=True)

    # Print a quick preview of the raw granular data
    print("\nPreview of all individual gene metrics across window distances:")
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(full_metrics_df[['cell_type', 'distance', 'gene_name', 
                               'aupr_delta_vs_label', 'auroc_delta_vs_label', 
                               'n_total_peaks', 'n_positive_peaks']].round(4).head(10))

    # Save the complete detailed benchmark file globally without mathematical compression
    output_filename = os.path.join(label_save_dir, f"nearby_aupr_{ct}_permutation.csv")
    full_metrics_df.to_csv(output_filename, index=False)
    print(f"\n[Success] Detailed structural file saved at: {output_filename}")
else:
    print("Execution complete: No data aligned across any selected gene window.")