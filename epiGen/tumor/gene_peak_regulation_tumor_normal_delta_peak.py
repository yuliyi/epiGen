import os
import anndata as ad
import numpy as np
import pandas as pd
from pyranges import read_gtf
from sklearn.mixture import GaussianMixture
from utils import parse_args, get_genes_from_gencode
import warnings
warnings.filterwarnings('ignore')


def calculate_delta_aupr_for_celltype(delta_adata, gene_name, gene_chrom, gene_start, gene_end,
                                      celltype_key='type', celltypes=['Tumor', 'Normal'], distances=[5e5, 3e5, 1e5]):
    """
    Extract significant gene-peak links dynamically per cell type based on perturbation scores.
    Crucial Fix: Thresholding is now strictly intra-cell-type to ensure downstream Motif purity.
    """
    peaks_df = delta_adata.var_names.to_series().str.split('-', expand=True)
    peak_chroms = peaks_df[0].values
    peak_starts = peaks_df[1].astype(int).values
    peak_ends = peaks_df[2].astype(int).values

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

            # 4. Extract significant links where delta > threshold
            high_mask = delta_window > threshold
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
                    'threshold_used': threshold # Added for transparency
                })

    return pd.DataFrame(link_results)


args = parse_args()
print(args)
tis = args.tis
ct = args.ct
data_dir = args.data_dir
obs_tissue_dict = {'CE336E1-S1': 'uterus', 'CE354E1-S1': 'uterus', 'CE357E1-S1': 'uterus', 'HT235B1-S1H1': 'breast', 'HT243B1-S1H4': 'breast', 'HT263B1-S1H1': 'breast', 'HT297B1-S1H1': 'breast', 'HT305B1-S1H1': 'breast', 'HT497B1-S1H1': 'breast', 'HT514B1-S1H3': 'breast', 'HT545B1-S1H1': 'breast', 'CE346E1-S1': 'uterus', 'CE347E1-S1K1': 'uterus', 'CE348E1-S1K1': 'uterus', 'CE349E1-S1': 'uterus', 'CE352E1-S1K1': 'uterus', 'CE356E1-S1': 'uterus', 'CM268C1-S1': 'colon', 'CM268C1-T1': 'colon', 'HT230C1-TH1': 'colon', 'HT250C1-TH1K1': 'colon', 'HT253C1-TH1K1': 'colon', 'HT291C1-M1A3': 'colon', 'HT307C1-TH1K1': 'colon', 'SP369H1-MC1': 'colon', 'HTX003S1-S1A1': 'tongue', 'HTX012S1-S1': 'tongue', 'HTX013S1-S1': 'tongue', 'P5216-N1': 'tongue', 'P5216-N2': 'tongue', 'P5504-N1': 'tongue', 'P5532-N1': 'tongue', 'P5576-1N1': 'tongue', 'P5590-N1': 'tongue', 'HT090P1-T2A3': 'pancreas', 'HT113P1-T2A3': 'pancreas', 'HT181P1-T1A3': 'pancreas', 'HT224P1-S1': 'pancreas', 'HT231P1-S1H3': 'pancreas', 'HT232P1-S1H1': 'pancreas', 'HT242P1-S1H1': 'pancreas', 'HT259P1-S1H1': 'pancreas', 'HT264P1-S1H2': 'pancreas', 'HT270P1-S1H2': 'pancreas', 'HT270P2-TH1FC1': 'pancreas', 'HT288P1-S1H4': 'pancreas', 'HT306P1-S1H1': 'pancreas', 'HT341P1-S1H1': 'pancreas', 'HT390P1-S1H1': 'pancreas', 'HT412P1-S1H1': 'pancreas', 'HT447P1-TH1K1A3': 'pancreas', 'HT452P1-TH1K1': 'pancreas', 'ML1199M1-TY1': 'skin', 'ML1199M1-TY2': 'skin', 'ML1232M1-TY1': 'skin', 'ML1239M1-TF1': 'skin', 'ML1294M1-TY1': 'skin', 'ML1332M1-TY1': 'skin', 'ML1511M1-S1': 'skin', 'ML1525M1-TY1': 'skin', 'ML1526M1-TA1': 'skin', 'CPT1541DU-S1': 'uterus', 'CPT1541DU-T1': 'uterus', 'CPT2373DU-S1': 'uterus', 'CPT2373DU-T1': 'uterus', 'CPT704DU-M1': 'uterus', 'CPT704DU-S1': 'uterus', 'CPT704DU-T1': 'uterus'}
celltypes = ['Tumor', 'Normal']
distances=[5e5]
celltype_key='type'
L = 500000
top = 50
bin_thres = 98
gene_range_df = get_genes_from_gencode(os.path.join(data_dir, "gencode.v47.annotation.gtf"))
link_results = pd.DataFrame()
gene_pd = pd.read_csv(f"{data_dir}/perturbation_tumor/diff_exp_tumor_normal_gene_names_{tis}.csv")

for gene_name in gene_pd[ct].dropna().values[:top]:
    if not os.path.exists(f"{data_dir}/perturbation_tumor/gene_peak_links/{tis}_{ct}_{gene_name}_nearby_links.csv"):
        gene_info = gene_range_df[gene_range_df['name'] == gene_name]
        gene_chrom = gene_info['chrom'].values[0]
        gene_s = gene_info['start'].values[0]
        gene_e = gene_info['end'].values[0]
        gene_start = gene_s if gene_s < gene_e else gene_e
        gene_end = gene_e if gene_s < gene_e else gene_s
        search_start, search_end = gene_start - L, gene_end + L
        print(f"Analyzing tissue: {tis}, gene: {gene_name}, chrom: {gene_chrom}, gene_start: {gene_start}, gene_end: {gene_end}")

        gene_atac_delta_adata = ad.read_h5ad(data_dir + f"/perturbation_tumor/gene_peak_regulation/atac_delta_tumor_normal_{ct}_{gene_name}_{gene_chrom}.h5ad")
        gene_atac_delta_adata.obs['tissue'] = [obs_tissue_dict[x] for x in gene_atac_delta_adata.obs.batch.values]
        gene_atac_delta_adata.obs[celltype_key] = np.where(gene_atac_delta_adata.obs.cell_type == 'Tumor', 'Tumor', 'Normal')

        peaks = [
            peak for peak in gene_atac_delta_adata.var_names 
            if (lambda p: (p[0]==gene_chrom and int(p[2])>=search_start and int(p[1])<=search_end))
            (peak.split('-'))
        ]
        gene_atac_delta_adata = gene_atac_delta_adata[gene_atac_delta_adata.obs.tissue == tis, peaks]
        # Compute strictly intra-cell-type links
        links = calculate_delta_aupr_for_celltype(
            gene_atac_delta_adata, gene_name, gene_chrom, gene_start, gene_end, celltype_key, celltypes, distances
        )
        del gene_atac_delta_adata

        links.to_csv(data_dir + f"/perturbation_tumor/gene_peak_links/{tis}_{ct}_{gene_name}_nearby_links.csv")
    else:
        links = pd.read_csv(data_dir + f"/perturbation_tumor/gene_peak_links/{tis}_{ct}_{gene_name}_nearby_links.csv", index_col=0)
    link_results = pd.concat([link_results, links], ignore_index=True)

link_results.to_csv(data_dir + f"/perturbation_tumor/gene_peak_links/{tis}_{ct}_nearby_links.csv")