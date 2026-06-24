## Compute perturbation effect as a function of distance to gene TSS ## 
import os
import copy
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import genomic_features as gf
import bioframe
import warnings
warnings.filterwarnings('ignore')
import random
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from epiGen_models import load_pretrain, sparse_scipy_to_tensor, ATACformer
from utils import parse_args, load_model, evaluate_perturbation, finetune_batch_embedding, simulate_perturbation
args = parse_args()


def get_TSS(gene_ranges_df: pd.DataFrame, perturbed_gene: str):
    '''
    Get position of TSS of gene of interest
    '''
    gene_range = gene_ranges_df.loc[perturbed_gene]
    if gene_range['seq_strand'] > 0:
        tss = gene_range['gene_seq_start']
    else:
        tss = gene_range['gene_seq_end']
    tss_range = gene_range.copy()
    tss_range['gene_seq_start'] = tss
    tss_range['gene_seq_end'] = tss+1
    tss_range['seq_strand'] = 1
    tss_range['gene_name'] = tss_range.name
    
    # Convert to bioframe compatible format (chrom, start, end, name)
    tss_range = pd.DataFrame(tss_range[['seq_name', 'gene_seq_start', 'gene_seq_end', 'gene_name']]).T
    tss_range.columns = ['chrom', 'start', 'end', 'name']
    tss_range['chrom'] = 'chr' + tss_range['chrom']
    return(tss_range.convert_dtypes())


def get_close_peaks_effect(atac_delta_adata: ad.AnnData, gene_ranges_df: pd.DataFrame, perturbed_gene: str, perturbed_gene_name: str, chr: str) -> None:
    '''
    Get perturbation effect at peaks around TSS
    '''
    ## Get position of promoter of perturbed gene
    tss_range = get_TSS(gene_ranges_df, perturbed_gene)

    ## Get peaks within 10kb of TSS and distance
    window = 10000
    close_peaks_df = bioframe.closest(peaks_df[peaks_df['chrom'] == tss_range.chrom[0]], tss_range)[['chrom', 'start', 'end', 'name', 'distance']]
    # close_peaks_df = close_peaks_df[close_peaks_df['distance'] < window]
    close_peaks_df = close_peaks_df[close_peaks_df['distance'] < window]

    ## Store mean perturbation delta for close peaks
    close_peaks_df['mean_delta'] =  np.abs(atac_delta_adata[:, close_peaks_df.name].X).mean(0)
    close_peaks_df['se_delta'] = np.abs(atac_delta_adata[:, close_peaks_df.name].X).std(0) / np.sqrt(close_peaks_df.shape[0])
    close_peaks_df['perturbed_gene'] = perturbed_gene
    close_peaks_df.to_csv(prefix + f'/TSS/TSS_perturb_{chr}_{perturbed_gene}_{perturbed_gene_name}.csv')


# seed
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Device configuration
device = torch.device("cuda:3")
prefix = args.data_dir + f"/perturbation"
# Load RNA expression data and process batch information
rna_list = ad.read_h5ad(args.data_dir + f"/rna_data/rna_independence_GSE234521.h5ad")
batch_info = rna_list.obs.batch.tolist()
batch_int, uniques = pd.factorize(batch_info)
# Shift batch indices to avoid overlapping with pre-trained batch IDs (assumes 222 old batches)
batch_int = np.array(list(map(lambda x: x + 222, batch_int))).astype(int)
print(f"batch size: {len(uniques)}")
mapping = dict(zip(uniques, range(222, 222+len(uniques))))
print(f"batch mapping: {mapping}")

sc.pp.normalize_total(rna_list, 1e4)
gene_list = rna_list.var_names.tolist()
all_perturb_genes = rna_list.var_names
all_perturb_gene_names = rna_list.var['gene_name']
# Fetch Ensembl gene annotations (GRCh38/hg38) using genomic_features
ensdb = gf.ensembl.annotation(species="Hsapiens", version="108")
gene_ranges_df = ensdb.genes(filter=gf.filters.EmptyFilter())
try:
    gene_ranges_df = gene_ranges_df[gene_ranges_df['gene_id'].isin(all_perturb_genes)].copy()
    gene_ranges_df = gene_ranges_df.set_index("gene_id")
except KeyError:
    gene_ranges_df = gene_ranges_df.set_index("gene_name")

# Load the foundational pre-trained RNA model (CellPLM)
cell_num = len(rna_list)
PRETRAIN_VERSION = '20230926_85M'
cellplm = load_pretrain(PRETRAIN_VERSION, {'head_type': 'embedder', 'mask_node_rate': 0}, args.save_dir)
print("perturb gene size:", len(all_perturb_genes))
# Prepare a subset (10%) of data to fine-tune the new batch embeddings
rna_batch = sparse_scipy_to_tensor(csr_matrix(rna_list.X.astype(float)))
_, ff_idx = train_test_split(range(len(rna_list)), test_size=0.1, random_state=args.seed)
ffset = torch.utils.data.TensorDataset(torch.IntTensor(ff_idx))
_ff = torch.utils.data.DataLoader(ffset, batch_size=args.batch_size, num_workers=4)

chrom_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
# Iterate over each chromosome for downstream analysis
for chr in chrom_list:
    chr = 'chr' + chr
    # Load peak BED file for the current chromosome
    peaks_df = pd.read_csv(args.data_dir + f'/atac_data/peaks_{chr}.bed', sep='\t', header=None)
    peaks_df.columns = ['chrom', 'start', 'end']
    peaks_df.loc[:,['start', 'end']] = peaks_df.loc[:,['start', 'end']].astype(int)
    peaks_df['name'] = peaks_df[['chrom', 'start', 'end']].astype(str).apply(lambda x: "-".join(x), axis=1)
    peak_count = len(peaks_df)
    # Initialize the cross-omics model
    model = ATACformer(args, cellplm, gene_list, peak_size=peak_count)
    model = load_model(model, args, chr).to(device)
    model = finetune_batch_embedding(model, len(uniques), _ff, rna_batch, batch_int, device, args.lr*0.1)
    for param in model.parameters():
        param.requires_grad = False
    print(f"****************perturb chromosome {chr} *******************")
    # Iterate through all genes to perform the in silico perturbation
    for i, g in enumerate(all_perturb_genes):
        g_name = all_perturb_gene_names[i]
        # filter gene located in chr
        if gene_ranges_df.loc[g, 'seq_name'] != chr[3:]:
            continue
        print(f"****************perturb gene {g} name {g_name} in {chr}******No.", i+1)
        # Skip if perturbation results have already been computed and saved
        if os.path.exists(prefix + f'/TSS/TSS_perturb_{chr}_{g}_{g_name}.csv'):
            continue
        # Select only the cells where the target gene is originally expressed (non-zero)
        indices_of_interest_gene = np.nonzero(rna_list[:, g].X.toarray())[0]
        print("indices_of_interest_gene num:", len(indices_of_interest_gene))
        if len(indices_of_interest_gene) == 0:
            continue
        # Extract subset of cells for testing
        rna_test_mat = rna_list[indices_of_interest_gene].copy()
        rna_test_original = sparse_scipy_to_tensor(csr_matrix(rna_test_mat.X.astype(float)))
        batch_test = torch.tensor(batch_int[indices_of_interest_gene], dtype=torch.int).flatten()
        testset = torch.utils.data.TensorDataset(torch.IntTensor(range(len(rna_test_mat))))
        _test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)
        
        # raw prediction without perturbation
        model_g = copy.deepcopy(model).to(device)
        for param in model_g.parameters():
            param.requires_grad = False
        model_g.eval()
        print("load model")
        # Generate reference ATAC peak predictions without perturbation
        predictions_raw = evaluate_perturbation(model_g, _test, rna_test_original, batch_test, device)
        print("predictions_perturb shape:", predictions_raw.shape)
        torch.cuda.empty_cache()
        # Predict perturbation
        predictions_perturb = simulate_perturbation(model_g, _test, rna_test_mat, rna_test_original, batch_test, g, gene_list, device, args.lr)
        print("predictions_perturb shape:", predictions_perturb.shape)
        # Calculate the delta between the perturbed predictions and the raw predictions
        predicted_changes = predictions_perturb - predictions_raw
        atac_delta_adata = ad.AnnData(X = predicted_changes, obs = rna_list.obs.iloc[indices_of_interest_gene])
        atac_delta_adata.var_names = peaks_df['name'].values
        get_close_peaks_effect(atac_delta_adata, gene_ranges_df, g, g_name, chr)
        del model_g
        torch.cuda.empty_cache()