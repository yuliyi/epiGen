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
from pyranges import read_gtf
from sklearn.model_selection import train_test_split
from epiGen_models import expand_batch_embedding, load_pretrain, sparse_scipy_to_tensor, ATACformer
from utils import parse_args, load_model, save_model, get_genes_from_gencode, evaluate_perturbation, finetune_batch_embedding, simulate_perturbation
    

def main(args, ct, chr, gene_start, gene_end, device):
    data_dir = args.data_dir
    model_dir = args.save_dir
    gene_name = args.gene_name
    
    # 1. Load and format ATAC peaks reference
    peaks_df = pd.read_csv(args.data_dir + f'/atac_data/peaks.bed', sep='\t', header=None)
    peaks_df.columns = ['chrom', 'start', 'end']
    peaks_df['start'] = peaks_df['start'].astype(int)
    peaks_df['end'] = peaks_df['end'].astype(int)
    peaks_df['name'] = peaks_df[['chrom', 'start', 'end']].astype(str).apply(lambda x: "-".join(x), axis=1)

    # 2. Load RNA feature references and create mapping
    gene_id_list = np.loadtxt(data_dir + '/rna_data/rna_cellplm_genes.txt', dtype=str)
    gene_name_list = np.loadtxt(data_dir + '/rna_data/rna_cellplm_gene_names.txt', dtype=str)
    rna_ref_df = dict(zip(gene_name_list, gene_id_list))

    # 3. Load RNA data and process batch information
    rna_list = ad.read_h5ad(data_dir + "/tumor/rna_tumor.h5ad")
    batch_info = rna_list.obs.batch.tolist()
    batch_int, uniques = pd.factorize(batch_info)
    batch_int = torch.from_numpy(np.array(list(map(lambda x: x + 222, batch_int))).astype(int))
    print(f"batch size: {len(uniques)}")
    mapping = dict(zip(uniques, range(222, 222+len(uniques))))
    print(f"batch mapping: {mapping}")

    # 4. Normalize RNA data
    sc.pp.normalize_total(rna_list, 1e4)
    gene_list = rna_list.var_names.tolist()

    gene_ensembl = rna_ref_df[gene_name]

    # 5. Initialize CellPLM (Encoder) and ATACformer (Decoder)
    PRETRAIN_VERSION = '20230926_85M'
    cellplm = load_pretrain(PRETRAIN_VERSION, {'head_type': 'embedder', 'mask_node_rate': 0}, model_dir)
    
    print("chrom:", chr)
    peaks_chr = peaks_df.loc[peaks_df['chrom'] == chr]
    peak_count = len(peaks_chr)
    
    # Initialize full model
    model = ATACformer(args, cellplm, gene_list, peak_size=peak_count)
    model_state_file = os.path.join(args.save_dir, f'model_fold_{args.fold}')
    
    # 6. Fine-tune or load batch embeddings
    if os.path.exists(os.path.join(model_state_file, f'{args.chr}_decoder_{args.fold}_expand_embed.pt')):
        # Load directly if fine-tuning was already completed for this fold/chromosome
        model.batch_embed = expand_batch_embedding(model.batch_embed, num_old_batches=222, num_new_batches=len(uniques))
        model = load_model(model, args, False).to(device)
    else:
        # Prepare dataset for fine-tuning batch embeddings
        rna_batch = sparse_scipy_to_tensor(csr_matrix(rna_list.X.astype(float)))
        _, ff_idx = train_test_split(range(len(rna_list)), test_size=0.1, random_state=args.seed)
        ffset = torch.utils.data.TensorDataset(torch.IntTensor(ff_idx))
        _ff = torch.utils.data.DataLoader(ffset, batch_size=args.batch_size, num_workers=4)

        # Load pre-trained model and fine-tune the expanded batch embeddings
        model = load_model(model, args).to(device)
        model = finetune_batch_embedding(model, len(uniques), _ff, rna_batch, batch_int, device, args.lr*0.1)
        save_model(model, args)
    # Ensure decoder parameters are frozen for the upcoming in-silico perturbation
    for param in model.parameters():
        param.requires_grad = False

    print(f"****************perturb gene {gene_name}")
    
    # 7. Identify target cells (only cells where the gene is originally expressed)
    indices_of_interest_gene = np.nonzero(rna_list[:, gene_ensembl].X.toarray())[0]  
    if len(indices_of_interest_gene) == 0:
        raise ValueError("No cells express the target gene for perturbation.")
    else:
        print("indices_of_interest_gene:", len(indices_of_interest_gene))
    # Isolate data specific to the target cells
    rna_test_mat = rna_list[indices_of_interest_gene].copy()
    rna_test_original = sparse_scipy_to_tensor(csr_matrix(rna_test_mat.X.astype(float)))
    batch_int_test = batch_int[indices_of_interest_gene]
    testset = torch.utils.data.TensorDataset(torch.IntTensor(range(len(rna_test_mat))))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)

    # 8. Baseline ATAC predictions (Raw predictions without permutation)
    print("load model")
    predictions_raw = evaluate_perturbation(model, _test, rna_test_original, batch_int_test, device)
    print("predictions_raw shape:", predictions_raw.shape)
    torch.cuda.empty_cache()

    # 9. Perform In-silico Permutation (Null Distribution Generation)
    predictions_perturb = simulate_perturbation(model, _test, rna_test_mat, rna_test_original, batch_int_test, gene_ensembl, gene_list, device, lr=args.lr, permutation=True)
    print("predictions_perturb shape:", predictions_perturb.shape)
    
    predicted_changes = predictions_perturb - predictions_raw  # Calculate δX (Permuted - Raw)
    del predictions_perturb, predictions_raw
    
    L = 500000
    search_start = gene_start - L
    search_end = gene_end + L
    # Boolean mask: Select peaks where the region overlaps the search window
    window_mask = (peaks_chr['end'] >= search_start) & (peaks_chr['start'] <= search_end)
    peaks_nearby = peaks_chr[window_mask]
    predicted_changes_nearby = predicted_changes[:, window_mask.values]
    print(f"Filtered nearby peaks shape: {predicted_changes_nearby.shape}")
    atac_delta_adata = ad.AnnData(X=predicted_changes_nearby, obs=rna_list.obs.iloc[indices_of_interest_gene])
    atac_delta_adata.var_names = peaks_nearby['name'].values
    atac_delta_adata.write_h5ad(data_dir + f"/perturbation_tumor/gene_peak_regulation_permutation/atac_delta_tumor_normal_{ct}_{gene_name}_{chr}_permutation.h5ad", compression='gzip')
    
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    print("gpu_num:", gpu_num)
    args = parse_args()
    print(args)
    
    # Set seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    tis = args.tis
    ct = args.ct
    device = torch.device(f"cuda:{args.cuda}")
    gene_range_df = get_genes_from_gencode(os.path.join(args.data_dir, "gencode.v47.annotation.gtf"))
    gene_pd = pd.read_csv(f"{args.data_dir}/perturbation_tumor/diff_exp_tumor_normal_gene_names_{tis}.csv")
    gene_name = args.gene_name
    gene_info = gene_range_df[gene_range_df['name'] == gene_name]
    gene_chrom = gene_info['chrom'].values[0]
    args.chr = gene_chrom

    gene_s = gene_info['start'].values[0]
    gene_e = gene_info['end'].values[0]
    gene_start = gene_s if gene_s < gene_e else gene_e
    gene_end = gene_e if gene_s < gene_e else gene_s
    if not os.path.exists(f"{args.data_dir}/perturbation_tumor/gene_peak_regulation_permutation/atac_delta_tumor_normal_{ct}_{gene_name}_{gene_chrom}_permutation.h5ad"):
        main(args, ct, gene_chrom, gene_start, gene_end, device)