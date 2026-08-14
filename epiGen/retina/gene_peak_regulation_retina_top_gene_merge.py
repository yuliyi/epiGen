import os
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, f1_score
from utils import parse_cal_label_args, tfidf_seurat, top_k_metrics, get_genes_from_gencode, calculate_celltype_specific_correlation_with_labels
import warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture


def merge_fragment(adata, bin_size=10000):
    """
    Merge raw ATAC peaks into larger fixed-size genomic bins (e.g., 10kb) via matrix multiplication.
    """
    coords = adata.var_names.to_series().str.extract(r'(?P<chrom>chr[^_:-]+)[_:-](?P<start>\d+)[_:-](?P<end>\d+)')
    if coords.isna().any().any():
        print("warning: peaks merge failed")

    coords['start'] = coords['start'].astype(int)
    coords['bin_start'] = (coords['start'] // bin_size) * bin_size
    coords['bin_end'] = coords['bin_start'] + bin_size
    coords['new_region'] = coords['chrom'] + '-' + coords['bin_start'].astype(str) + '-' + coords['bin_end'].astype(str)

    new_regions = pd.Categorical(coords['new_region'])

    # Mapping matrix framework construction (shape: raw_peak × new_bin)
    row_indices = np.arange(adata.n_vars) 
    col_indices = new_regions.codes     
    data = np.ones(adata.n_vars)

    mapping_matrix = sp.coo_matrix(
        (data, (row_indices, col_indices)), 
        shape=(adata.n_vars, len(new_regions.categories))
    ).tocsc()

    # Sparse matrix multiplication execution
    new_X = adata.X @ mapping_matrix
    new_X.data = np.ones_like(new_X.data, dtype=np.int8)  # Binarize cell-level accessibility
    new_var = pd.DataFrame(index=new_regions.categories)
    adata_bin = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)

    print(f"Pre-merge adata shape: {adata.shape}")
    print(f"Post-merge (10kb Binned) adata shape: {adata_bin.shape}")
    return adata_bin


def filter_bin_chrom(delta_adata, pred_correlations, labels, pred_labels, gene_name, gene_chrom, gene_start, gene_end,
                     ct='RGCs', distances=[5e5, 4e5, 3e5, 2e5, 1e5], bin_size=10000):
    """
    Map raw perturbation peaks into 10kb genomic bins, apply local binarization via permutation background,
    perform Max-Pooling aggregation.
    """
    aligned_data = {}
    
    # 1. Cell-level aggregation to obtain continuous continuous activity scores
    delta_values = np.abs(delta_adata.X).mean(axis=0)

    # 2. Extract genomic coordinates and compute 10kb target bins
    coords = delta_adata.var_names.to_series().str.extract(r'(?P<chrom>chr[^_:-]+)[_:-](?P<start>\d+)[_:-](?P<end>\d+)')
    coords['start'] = coords['start'].astype(int)
    coords['end'] = coords['end'].astype(int)
    coords['bin_start'] = (coords['start'] // bin_size) * bin_size
    coords['bin_end'] = coords['bin_start'] + bin_size
    coords['new_region'] = coords['chrom'] + '-' + coords['bin_start'].astype(str) + '-' + coords['bin_end'].astype(str)
    
    # Create peak-level base dataframe (is_tf_bound is completely removed from here)
    df_base = pd.DataFrame({
        'chrom': coords['chrom'].values,
        'start': coords['start'].values,
        'end': coords['end'].values,
        'delta': delta_values,
        'bin': coords['new_region'].values
    })

    if ct in labels.columns:
        ct_labels = labels[ct]
        ct_pred_labels = pred_labels[ct]
        ct_pred_correlations = pred_correlations[ct]
        
        for dis in distances:
            search_start = gene_start - dis
            search_end = gene_end + dis
            
            # Filter rows located strictly within the spatial genomic distance window
            mask_local = (df_base['chrom'] == gene_chrom) & \
                         (df_base['start'] <= search_end) & \
                         (df_base['end'] >= search_start)
            df_local = df_base[mask_local].copy()

            if df_local.empty:
                aligned_data[str(dis)] = pd.DataFrame()
                continue

            # Use GMM to set binarization threshold
            X = df_local['delta'].values.reshape(-1, 1)
            try:
                best_gmm = GaussianMixture(n_components=3, covariance_type='tied', random_state=42).fit(X)
                means = best_gmm.means_.flatten()
                signal_class = np.argmax(means)
                gmm_labels = best_gmm.predict(X)
                df_local['delta_binary'] = (gmm_labels == signal_class).astype(int)
            except Exception as e:
                print(f"GMM fitting failed for distance {dis}: {e}. Fallback to percentile.")
                threshold = np.percentile(df_local['delta'], bin_thres)
                df_local['delta_binary'] = np.where(df_local['delta'] > threshold, 1, 0).astype(np.int32)

            # Max Pooling strategy: Aggregate continuous signal and binary calls into 10kb bins
            df_max = df_local.groupby('bin').agg({
                'delta': 'max',
                'delta_binary': 'max'
            })

            # Fetch the precise candidate target bins from Ground Truth labels matching the window
            target_bins = [
                b for b in labels.index
                if b.split('-')[0] == gene_chrom and int(b.split('-')[2]) >= search_start and int(b.split('-')[1]) <= search_end
            ]
            
            if not target_bins:
                aligned_data[str(dis)] = pd.DataFrame()
                continue

            # Reindex and perform conservative zero-filling for missing regions
            delta_series = df_max['delta'].reindex(target_bins).fillna(0.0)
            delta_binary_series = df_max['delta_binary'].reindex(target_bins).fillna(0).astype(int)
            
            # Align Ground Truth labels with mathematical predictions
            df_aligned = pd.DataFrame({
                'gene_name': [gene_name] * len(target_bins),
                'label': ct_labels[target_bins].values,
                'delta': delta_series.values,
                'delta_binary': delta_binary_series.values,
                'pred_labels': ct_pred_labels[target_bins].values,
                'preds': ct_pred_correlations[target_bins].values,
            }, index=target_bins)
            
            aligned_data[str(dis)] = df_aligned
            
    return aligned_data


def calculate_delta_aupr_for_single_gene(aligned_df, gene_name, target_ct='RGCs'):
    """
    Compute classification metrics (AUPRC, AUROC, MCC, F1) for a single gene's binned dataframe.
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
bin_size = 10000

# 2. Load Datasets
rna_list = ad.read_h5ad(os.path.join(data_dir, "rna_data/Multi_Fetal_19W4d_FR_rna_ct.h5ad"))
atac_list = ad.read_h5ad(os.path.join(data_dir, "atac_data/Multi_Fetal_19W4d_FR_atac_cpeak.h5ad"))
ct_map = {'RGCs': ['midget ganglion cell of retina', 'retinal ganglion cell', 'ON parasol ganglion cell', 'OFF parasol ganglion cell'],
          'RPCs': ['retinal progenitor cell'],
          'MCs': ['Mueller cell'],
          'ACs': ['GABAergic amacrine cell', 'amacrine cell', 'glycinergic amacrine cell', 'starburst amacrine cell'],
          'BCs': ['retinal bipolar neuron', 'diffuse bipolar 2 cell', 'diffuse bipolar 3b cell', 'diffuse bipolar 1 cell', 'diffuse bipolar 3a cell', 'rod bipolar cell', 'ON-bipolar cell', 'diffuse bipolar 6 cell', 'diffuse bipolar 4 cell'],
          'PCs': ['retinal rod cell', 'retinal cone cell', 'S cone cell']}

celltype_key = 'Cluster'
distances = [5e5, 4e5, 3e5, 2e5, 1e5]
L = 500000
top = 10
pval_threshold = 0.05
corr_threshold = 0.01
min_pct = 0.05
bin_thres = 98
all_types = [item for sublist in ct_map.values() for item in sublist]
rna_list_mat = rna_list[rna_list.obs.cell_type.isin(all_types)].copy()

# Apply 10kb Bin Merge to global ATAC first, then perform Seurat TF-IDF normalization
print("Executing Global ATAC Peak-to-Bin Downsampling...")
atac_list_bin = merge_fragment(atac_list, bin_size)
atac_list_bin.X = tfidf_seurat(atac_list_bin)

rna_st = ad.read_h5ad(data_dir + f"/rna_data/rna_retina_st.h5ad")
atac_pred = ad.read_h5ad(os.path.join(data_dir, "prediction/retina/allchr_retina_bins.h5ad"))
atac_pred.obs = rna_st.obs.copy()
atac_pred_bin = merge_fragment(atac_pred, bin_size)
atac_pred_bin.X = tfidf_seurat(atac_pred_bin)
rna_st = rna_st[rna_st.obs.Cluster.isin(ct_map.keys())].copy()
atac_pred_bin = atac_pred_bin[atac_pred_bin.obs.Cluster.isin(ct_map.keys())].copy()

# Load annotations
gene_range_df = get_genes_from_gencode(os.path.join(data_dir, "gencode.v47.annotation.gtf"))
gene_pd = pd.read_csv(os.path.join(data_dir, "perturbation_retina/diff_exp_gene_names.csv"))

# Setup output folders for 10kb regions database
label_save_dir = os.path.join(data_dir, "perturbation_retina/gene_peak_nearby_merge")
os.makedirs(label_save_dir, exist_ok=True) 

distance_results_collector = {str(dis): [] for dis in distances}

# 3. Main Evaluation Loop: Iterating through Top 10 DEGs
for gene_name in gene_pd[ct].dropna().values[:top]:
    gene_info = gene_range_df[gene_range_df['name'] == gene_name]
    if gene_info.empty:
        for dis in distances:
            distance_results_collector[str(dis)].append(calculate_delta_aupr_for_single_gene(None, gene_name, target_ct=ct))
        print(f"Skipping {gene_name}: not found in gene annotations.")
        continue
        
    gene_chrom = gene_info['chrom'].values[0]
    gene_s = gene_info['start'].values[0]
    gene_e = gene_info['end'].values[0]
    gene_start = min(gene_s, gene_e)
    gene_end = max(gene_s, gene_e)
    search_start, search_end = gene_start - L, gene_end + L
    print(f"Analyzing celltype: {ct}, gene: {gene_name}, chrom: {gene_chrom}, bin_size: {bin_size}")

    # Load actual and permuted perturbation matrix outputs (stored at raw peak levels)
    delta_path = os.path.join(data_dir, f"perturbation_retina/gene_peak_regulation/atac_delta_retina_{ct}_{gene_name}_{gene_chrom}.h5ad")
    perm_path = os.path.join(data_dir, f"perturbation_retina/gene_peak_regulation_permutation/atac_delta_retina_{ct}_{gene_name}_{gene_chrom}_permutation.h5ad")
    
    if not (os.path.exists(delta_path) and os.path.exists(perm_path)):
        for dis in distances:
            distance_results_collector[str(dis)].append(calculate_delta_aupr_for_single_gene(None, gene_name, target_ct=ct))
        print(f"Skipping {gene_name}: Perturbation output files missing.")
        continue
        
    gene_atac_delta_adata = ad.read_h5ad(delta_path)

    # Filter out binned regions within 500kb window to optimize Spearman correlation computation
    peaks = [
        peak for peak in atac_list_bin.var_names 
        if peak.split('-')[0] == gene_chrom and int(peak.split('-')[2]) >= search_start and int(peak.split('-')[1]) <= search_end
    ]
    atac_list_mat = atac_list_bin[rna_list_mat.obs_names, peaks].copy()
    atac_pred_mat = atac_pred_bin[:, peaks].copy()
    
    # Filter continuous perturbation models based on available target cells
    mask_delta = gene_atac_delta_adata.obs[celltype_key] == ct
    gene_atac_delta_adata = gene_atac_delta_adata[mask_delta]
    
    # 4. Generate or Load Binned Ground Truth Labels
    label_path = os.path.join(label_save_dir, f"{gene_name}_{ct}_nearby_labels_{bin_size}.csv")
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

    pred_label_path = os.path.join(label_save_dir, f"{gene_name}_{ct}_nearby_pred_labels_{bin_size}.csv")
    if not os.path.exists(pred_label_path):
        pred_correlations, _, _, pred_labels = \
            calculate_celltype_specific_correlation_with_labels(
                rna_st, atac_pred_mat, ct_map, gene_name, 'gene_name', 'Cluster', ct,
                min_pct=min_pct, pval_threshold=pval_threshold,
                corr_threshold=corr_threshold
            )
        pred_correlations.to_csv(label_save_dir + f"/{gene_name}_{ct}_nearby_pred_correlations_{bin_size}.csv")
        pred_labels.to_csv(pred_label_path)
    else:
        pred_correlations = pd.read_csv(label_save_dir + f"/{gene_name}_{ct}_nearby_pred_correlations_{bin_size}.csv", index_col=0)
        pred_labels = pd.read_csv(pred_label_path, index_col=0)

    # 5. Run Max-Pooling Aggregation and Alignment
    aligned_data = filter_bin_chrom(
        gene_atac_delta_adata, pred_correlations, labels, pred_labels, gene_name, gene_chrom, gene_start, gene_end, ct, distances, bin_size
    )
    
    # 6. Accumulate Granular Metric Records
    for dis in distances:
        dis_str = str(dis)
        current_df = aligned_data.get(dis_str, pd.DataFrame())
        metrics = calculate_delta_aupr_for_single_gene(current_df, gene_name, target_ct=ct)
        distance_results_collector[dis_str].append(metrics)

# ==========================================
# 4. Long-Format Detailed Table Generation
# ==========================================
all_distance_dfs = []

for dis in distance_results_collector.keys():
    gene_list_for_dis = distance_results_collector[dis]
    if len(gene_list_for_dis) == 0:
        continue
        
    df_dis_individual = pd.DataFrame(gene_list_for_dis)
    df_dis_individual['distance'] = str(dis)
    df_dis_individual['cell_type'] = ct
    
    cols = ['cell_type', 'distance', 'gene_name'] + \
           [c for c in df_dis_individual.columns if c not in ['cell_type', 'distance', 'gene_name']]
    df_dis_individual = df_dis_individual[cols]
    
    all_distance_dfs.append(df_dis_individual)

if all_distance_dfs:
    full_metrics_df = pd.concat(all_distance_dfs, axis=0).reset_index(drop=True)

    print("\nPreview of binned granular evaluation records:")
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(full_metrics_df[['cell_type', 'distance', 'gene_name', 
                               'aupr_delta_vs_label', 'n_total_peaks', 'n_positive_peaks']].head(15))

    # Save to global results csv path
    output_filename = os.path.join(label_save_dir, f"nearby_aupr_{ct}_permutation_{bin_size}.csv")
    full_metrics_df.to_csv(output_filename, index=False)
    print(f"\n[Success] 100% complete binned metrics database written to: {output_filename}")
else:
    print("Execution complete: No data aggregated.")