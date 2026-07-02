import os
import anndata as ad
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from epiGen_models import load_pretrain, sparse_scipy_to_tensor, ATACformer
from utils import parse_args, load_model, get_genes_from_gencode, evaluate_perturbation, finetune_batch_embedding, simulate_perturbation


def main(args, ct, chr):
    data_dir = args.data_dir
    model_dir = args.save_dir
    gene_name = args.gene_name
    # Load and format the master peak reference file
    peaks_df = pd.read_csv(args.data_dir + f'/atac_data/peaks.bed', sep='\t', header=None)
    peaks_df.columns = ['chrom', 'start', 'end']
    peaks_df['name'] = peaks_df[['chrom', 'start', 'end']].astype(str).apply(lambda x: "-".join(x), axis=1)
    # Load mappings to translate between common gene names and Ensembl IDs
    gene_id_list = np.loadtxt(data_dir + '/rna_data/rna_cellplm_genes.txt', dtype=str)
    gene_name_list = np.loadtxt(data_dir + '/rna_data/rna_cellplm_gene_names.txt', dtype=str)
    rna_ref_df = dict(zip(gene_name_list, gene_id_list))
    # Load downstream scRNA-seq AnnData and associate cell type metadata
    rna_list = ad.read_h5ad(data_dir + "/rna_data/rna_independence.h5ad")
    celltype = pd.read_csv(os.path.join(data_dir + "/statistic/celltype.csv"), index_col=0)
    rna_list.obs = celltype
    batch_info = rna_list.obs.batch.tolist()
    batch_int, uniques = pd.factorize(batch_info)
    # Shift batch indices to avoid overlapping with pre-trained batch IDs (assumes 222 old batches)
    batch_int = torch.from_numpy(np.array(list(map(lambda x: x + 222, batch_int))).astype(int))
    print(f"batch size: {len(uniques)}")
    mapping = dict(zip(uniques, range(222, 222+len(uniques))))
    print(f"batch mapping: {mapping}")
    sc.pp.normalize_total(rna_list, 1e4)
    gene_list = rna_list.var_names.tolist()

    device = torch.device("cuda:1")
    gene_ensembl = rna_ref_df[gene_name]
    # Load CellPLM Foundation Model
    PRETRAIN_VERSION = '20230926_85M'
    cellplm = load_pretrain(PRETRAIN_VERSION, model_dir)
    # Sample 10% of the dataset to adapt the batch embeddings to the new domain
    rna_batch = sparse_scipy_to_tensor(csr_matrix(rna_list.X.astype(float)))
    _, ff_idx = train_test_split(range(len(rna_list)), test_size=0.1, random_state=args.seed)
    ffset = torch.utils.data.TensorDataset(torch.IntTensor(ff_idx))
    _ff = torch.utils.data.DataLoader(ffset, batch_size=args.batch_size, num_workers=4)
    
    print("chrom:", chr)
    # Subset peaks strictly to the target chromosome
    peaks_chr = peaks_df.loc[peaks_df['chrom'] == chr]
    peak_count = len(peaks_chr)
    model = ATACformer(args, cellplm, gene_list, peak_size=peak_count)
    model = load_model(model, args, chr).to(device)
    model = finetune_batch_embedding(model, len(uniques), _ff, rna_batch, batch_int, device, args.lr*0.1)
    for param in model.parameters():
        param.requires_grad = False

    print(f"****************perturb gene {gene_name}")
    # Select only the cells where the target gene is originally expressed (non-zero)
    indices_of_interest_gene = np.nonzero(rna_list[:, gene_ensembl].X.toarray())[0]  # record non-zero cell index given gene in rna matrix
    if len(indices_of_interest_gene) == 0:
        raise ValueError("no cell for perturbation")
    else:
        print("indices_of_interest_gene:", len(indices_of_interest_gene))
    # Extract subset of cells for testing
    rna_test_mat = rna_list[indices_of_interest_gene].copy()
    rna_test_original = sparse_scipy_to_tensor(csr_matrix(rna_test_mat.X.astype(float)))
    batch_test = torch.tensor(batch_int[indices_of_interest_gene], dtype=torch.int).flatten()
    testset = torch.utils.data.TensorDataset(torch.IntTensor(range(len(rna_test_mat))))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)

    print("load model")
    # Generate reference ATAC peak predictions without perturbation
    predictions_raw = evaluate_perturbation(model, _test, rna_test_original, batch_test, device)
    print("predictions_perturb shape:", predictions_raw.shape)
    torch.cuda.empty_cache()

    predictions_perturb = simulate_perturbation(model, _test, rna_test_mat, rna_test_original, batch_test, gene_ensembl, gene_list, device, lr=args.lr)
    print("predictions_perturb shape:", predictions_perturb.shape)
    # Calculate the delta between the perturbed predictions and the raw predictions
    predicted_changes = predictions_perturb - predictions_raw  # δX for non-zero cell
    atac_delta_adata = ad.AnnData(X = predicted_changes, obs = rna_list.obs.iloc[indices_of_interest_gene])
    atac_delta_adata.var_names = peaks_chr['name'].values
    atac_delta_adata.write_h5ad(data_dir + f"/perturbation/gene_peak_regulation/atac_delta_{ct}_{gene_name}_{chr}.h5ad", compression='gzip')
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    print("gpu_num:", gpu_num)
    args = parse_args()
    print(args)
    # seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ct = args.ct
    gene_range_df = get_genes_from_gencode(os.path.join(args.data_dir, "gencode.v47.annotation.gtf.gz"))
    gene_pd = pd.read_csv(f"{args.data_dir}/perturbation/diff_exp_gene_names.csv")
    for gene_name in gene_pd[ct].dropna().values[:20]:
        gene_info = gene_range_df[gene_range_df['name'] == gene_name]
        gene_chrom = gene_info['chrom'].values[0]
        args.gene_name = gene_name
        if not os.path.exists(f"{args.data_dir}/perturbation/gene_peak_regulation/atac_delta_{ct}_{gene_name}_{gene_chrom}.h5ad"):
            main(args, ct, gene_chrom)