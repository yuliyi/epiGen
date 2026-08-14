import os
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from utils import parse_args, get_genes_from_gencode
import warnings
warnings.filterwarnings('ignore')


def calculate_delta_aupr_for_celltype(delta_adata, gene_name, gene_chrom, gene_start, gene_end,
                                      celltype_key='type', celltypes=['RGCs', 'Others'], distances=[5e5, 3e5, 1e5]):
    """
    Extract significant gene-peak links dynamically per cell type based on perturbation scores.
    Crucial Fix: Thresholding is now strictly intra-cell-type to ensure downstream Motif purity.
    """
    peaks_df = delta_adata.var_names.str.split('-', expand=True)
    if isinstance(peaks_df, pd.MultiIndex):
        peak_chroms = peaks_df.get_level_values(0).values
        peak_starts = peaks_df.get_level_values(1).astype(int).values
        peak_ends = peaks_df.get_level_values(2).astype(int).values
    else:
        peak_chroms = peaks_df.values
        peak_starts = peaks_df.astype(int).values
        peak_ends = peaks_df.astype(int).values

    link_results = []

    for ct in celltypes:
        # 1. Filter AnnData for specific cell type FIRST
        ct_mask = delta_adata.obs[celltype_key] == ct
        if not ct_mask.any():
            continue
            
        adata_ct = delta_adata[ct_mask]

        # 2. Calculate cell-type specific Delta and Permutation background
        celltype_delta = np.abs(adata_ct.X.toarray()).mean(axis=0)

        # 3. Filter peaks per distance window using the specific cell type's background
        for dis in distances:
            search_start = gene_start - dis
            search_end = gene_end + dis
            
            # Find peaks within the genomic window
            dist_mask = (peak_chroms == gene_chrom) & (peak_ends >= search_start) & (peak_starts <= search_end)
            
            if not dist_mask.any():
                continue

            delta_window = celltype_delta[dist_mask]
            
            X = delta_window.reshape(-1, 1)
            try:
                best_gmm = GaussianMixture(n_components=3, covariance_type='tied', random_state=42).fit(X)
                means = best_gmm.means_.flatten()
                signal_class = np.argmax(means)
                gmm_labels = best_gmm.predict(X)
                third_cluster_values = delta_window[gmm_labels == signal_class]
                threshold = third_cluster_values.min()
            except Exception as e:
                print(f"GMM fitting failed for distance {dis}: {e}. Fallback to percentile.")
                threshold = np.percentile(delta_window, bin_thres)

            # 4. Extract significant links where delta >= threshold
            high_mask = delta_window >= threshold
            significant_indices = np.where(dist_mask)[0][high_mask]

            if len(significant_indices) == 0:
                continue

            peaks_names = delta_adata.var_names[significant_indices]
            delta_vals = celltype_delta[significant_indices]

            for peak_name, delta_val in zip(peaks_names, delta_vals):
                link_results.append({
                    'type': ct,
                    'dis': str(dis),
                    'gene_name': gene_name,
                    'peak': peak_name,
                    'delta': delta_val,
                    'threshold_used': threshold
                })

    return pd.DataFrame(link_results)


args = parse_args()
print(args)
ct = args.ct
data_dir = args.data_dir
celltypes = [ct, 'Others']
distances=[5e5]
celltype_key='type'
L = 500000
top = 50
bin_thres=98
gene_range_df = get_genes_from_gencode(os.path.join(data_dir, "gencode.v47.annotation.gtf"))
link_results = pd.DataFrame()
gene_pd = pd.read_csv(f"{data_dir}/perturbation_retina/diff_exp_gene_names.csv")

for gene_name in gene_pd[ct].dropna().values[:top]:
    if not os.path.exists(f"{data_dir}/perturbation_retina/gene_peak_links/{ct}_{gene_name}_nearby_links.csv"):
        gene_info = gene_range_df[gene_range_df['name'] == gene_name]
        gene_chrom = gene_info['chrom'].values[0]
        gene_s = gene_info['start'].values[0]
        gene_e = gene_info['end'].values[0]
        gene_start = gene_s if gene_s < gene_e else gene_e
        gene_end = gene_e if gene_s < gene_e else gene_s
        search_start, search_end = gene_start - L, gene_end + L
        print(f"Analyzing ct: {ct}, gene: {gene_name}, chrom: {gene_chrom}, gene_start: {gene_start}, gene_end: {gene_end}")

        gene_atac_delta_adata = ad.read_h5ad(data_dir + f"/perturbation_retina/gene_peak_regulation/atac_delta_retina_{ct}_{gene_name}_{gene_chrom}.h5ad")
        gene_atac_delta_adata.obs[celltype_key] = np.where(gene_atac_delta_adata.obs.Cluster == ct, ct, 'Others')

        peaks = [
            peak for peak in gene_atac_delta_adata.var_names 
            if (lambda p: (p[0]==gene_chrom and int(p[2])>=search_start and int(p[1])<=search_end))
            (peak.split('-'))
        ]
        gene_atac_delta_adata = gene_atac_delta_adata[:, peaks]
        # Compute strictly intra-cell-type links
        links = calculate_delta_aupr_for_celltype(
            gene_atac_delta_adata, gene_name, gene_chrom, gene_start, gene_end, celltype_key, celltypes, distances
        )
        del gene_atac_delta_adata

        links.to_csv(data_dir + f"/perturbation_retina/gene_peak_links/{ct}_{gene_name}_nearby_links.csv")
    else:
        links = pd.read_csv(data_dir + f"/perturbation_retina/gene_peak_links/{ct}_{gene_name}_nearby_links.csv", index_col=0)
    link_results = pd.concat([link_results, links], ignore_index=True)

link_results.to_csv(data_dir + f"/perturbation_retina/gene_peak_links/{ct}_nearby_links.csv")