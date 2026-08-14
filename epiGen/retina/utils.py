import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import argparse
import torch
import scipy
import numpy as np
import pandas as pd
import anndata as ad
import json
from CellPLM.model import OmicsFormer
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from pyranges import read_gtf
from epiGen_models import expand_batch_embedding, sparse_scipy_to_tensor, XDict

def parse_cal_label_args():
    """Parse command line arguments for parameters and paths."""
    parser = argparse.ArgumentParser(description="Run epiGen In-silico Perturbation Analysis.")
    parser.add_argument('--data_dir', default='/path/data',
                        help='Input data path.')
    parser.add_argument('--ct', type=str, default='RGCs', help='Target cell type.')
    args = parser.parse_args()
    return args


def parse_args():
    """Parse command line arguments for hyperparameters and paths."""
    parser = argparse.ArgumentParser(description="Run epiGen In-silico Perturbation.")
    parser.add_argument('--data_dir', default='/path/data', help='Input data path.')
    parser.add_argument('--save_dir', default='/path/model', help='Directory to save the trained model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Dimension of the batch embedding.')
    parser.add_argument('--n_epoch', type=int, default=500, help='Total number of training epochs.')
    parser.add_argument('--stopping_steps', type=int, default=100, help='Patience (number of epochs) for early stopping.')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training.')
    parser.add_argument('--chr', type=str, default='chr10', help='Specific chromosome to process.')
    parser.add_argument('--fold', type=str, default='retina', help='Name of the test dataset or cross-validation dataset.')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='Number of epochs for stage-1 warmup (decoder only).')
    parser.add_argument('--output_z', action="store_true", default=False, help='Flag to output the latent embeddings (z).')
    parser.add_argument('--mod', type=str, default='training', help='Execution mode: "training" or "testing".')
    parser.add_argument('--ct', type=str, default='RGCs', help='Target cell type.')
    parser.add_argument('--gene_name', type=str, default='', help='Target gene name for permutation.')
    parser.add_argument('--cuda', type=int, default=1, help='CUDA device ID.')
    args = parser.parse_args()
    return args


def load_pretrain(
        pretrain_prefix: str,
        overwrite_config: dict = None,
        pretrain_directory: str = './ckpt'):
    """Load the pre-trained foundation model CellPLM."""
    config_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.config.json')
    ckpt_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.best.ckpt')
    
    with open(config_path, "r") as openfile:
        config = json.load(openfile)
    config.update(overwrite_config)
    model = OmicsFormer(**config)
    
    pretrained_model_dict = torch.load(ckpt_path, weights_only=True)['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_model_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def load_model(model, args, is_pretrain=True):
    """Load model weights from either the pre-trained path or the fine-tuned fold path."""
    if is_pretrain:
        model_path = os.path.join(args.save_dir, f'model_fold_pretrain/{args.chr}_decoder_pretrain.pt')
    else:
        model_path = os.path.join(args.save_dir, f'model_fold_{args.fold}/{args.chr}_decoder_{args.fold}_expand_embed.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


def save_model(model, args):
    """Save model weights."""
    model_state_file = os.path.join(args.save_dir, f'model_fold_{args.fold}')
    torch.save(model.state_dict(), os.path.join(model_state_file, f'{args.chr}_decoder_{args.fold}_expand_embed.pt'))


def get_genes_from_gencode(file):
    """Extract protein-coding gene coordinates from a GTF annotation file."""
    gtf = read_gtf(file).as_df()
    gene_list = gtf[(gtf.Feature == 'gene') & (
        gtf.gene_type == 'protein_coding')][['Chromosome', 'Start', 'End', 'gene_name', 'gene_id']]
    gene_list.columns = ['chrom', 'start', 'end', 'name', 'id']
    return (gene_list.convert_dtypes())


def evaluate_perturbation(model, test_loader, rna_test, batch_test, device):
    model.eval()
    yy_score = []
    with torch.no_grad():
        for batch_cell in test_loader:
            batch_cell_idx = batch_cell[0].cpu()
            # Prepare input data dictionary with RNA sequence and batch information
            cell_rna = XDict({'x_seq': rna_test.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device),
                              'batch': batch_test[batch_cell_idx].to(device)})
            outs = model(cell_rna)[0]
            # Reshape and store predictions
            yy_score.append(outs.reshape(len(batch_cell_idx), model.peak_size).detach().cpu().numpy())
            del cell_rna, outs
            torch.cuda.empty_cache()
        a_preds = np.concatenate(yy_score, axis=0)
        del yy_score
    torch.cuda.empty_cache()
    return a_preds

    
def finetune_batch_embedding(model, num_new_batches, dataloader, rna_batch, batch_int, device, lr, epochs=50):
    """
    Fine-tunes the batch embedding layer of the model while freezing all other parameters.
    
    Args:
        model: The multi-omics model.
        num_new_batches (int): Number of new batches to add and fine-tune.
        dataloader (DataLoader): DataLoader yielding cell indices.
        rna_batch (Tensor): Sparse or dense tensor containing RNA sequences.
        batch_int (ndarray/Tensor): Array containing the batch index for each cell.
        rna_loss (Loss): Loss function used for RNA reconstruction.
        device (torch.device): Compute device.
        lr (float): Base learning rate.
        epochs (int): Number of epochs to train the batch embeddings.
        
    Returns:
        model: The model with updated and fine-tuned batch embeddings.
    """
    rna_loss = torch.nn.MSELoss()
    # Expand and fine-tune batch embeddings to adapt to the new dataset distribution
    model.batch_embed = expand_batch_embedding(model.batch_embed, num_old_batches=222, num_new_batches=num_new_batches).to(device)
    # Freeze all parameters except the newly expanded batch embedding layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.batch_embed.parameters():
        param.requires_grad = True
    optimizer_batch = torch.optim.AdamW(params=model.batch_embed.parameters(), lr=lr)
    model.eval()
    # Optimize batch embeddings for 50 epochs
    for epoch_batch in range(epochs):
        for batch_cell in dataloader:
            optimizer_batch.zero_grad()
            batch_cell_idx = batch_cell[0].cpu()
            cell_rna = XDict({'x_seq': rna_batch.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device),
                            'batch': batch_int[batch_cell_idx].to(device)})
            _, rna_, _ = model(cell_rna)
            ls = rna_loss(rna_.flatten(), torch.log1p(cell_rna['x_seq'].to_dense().flatten()))
            ls.backward()
            optimizer_batch.step()
    del optimizer_batch
    return model


def simulate_perturbation(model, test_loader, rna_test_mat, rna_test_original, batch_test, 
                                  gene_ensembl, gene_list, device, lr, permutation=False):
    """
    Simulates a gene knockout by optimizing the latent representation 'z' for 1 epoch,
    and returns the predicted ATAC accessibility based on the perturbed latent space.

    Parameters:
    -----------
    model : torch.nn.Module
        The pre-trained multi-omics model.
    test_loader : torch.utils.data.DataLoader
        DataLoader yielding batch indices.
    rna_test_mat : anndata.AnnData or similar array-like
        The RNA expression matrix for the cells of interest.
    rna_test_original : torch.sparse.Tensor
        The original, unperturbed RNA sparse tensor.
    batch_test : torch.Tensor
        Tensor containing batch indices for each cell.
    gene_ensembl : int or str
        The index or identifier of the target gene to be knocked out.
    gene_list : list
        List of gene names required by the RNA encoder.
    device : torch.device
        The device to run computations on.
    lr : float
        Learning rate for the latent space optimization. 

    Returns:
    --------
    predictions_perturb : np.ndarray
        The concatenated ATAC peak predictions after in silico perturbation.
    """
    
    rna_loss = torch.nn.MSELoss()
    # Set the expression of the target gene to 0 to simulate knockout
    if permutation:
        if isinstance(gene_ensembl, str):
            gene_col_idx = int(rna_test_mat.var_names.get_loc(gene_ensembl))
        else:
            gene_col_idx = int(gene_ensembl)
            
        gene_data_dense = rna_test_mat[:, gene_col_idx].X.toarray().flatten()
        group_col = rna_test_mat.obs['Cluster'].values
        # Intra-group shuffling to maintain cell-type specific expression distributions
        for group_name in np.unique(group_col):
            mask = group_col == group_name
            if mask.sum() > 0:
                gene_data_dense[mask] = np.random.permutation(gene_data_dense[mask])
        X_lil = rna_test_mat.X.tolil()
        X_lil[:, gene_col_idx] = gene_data_dense.reshape(-1, 1)
        rna_test_mat.X = X_lil.tocsr()
    else:
        rna_test_mat[:, gene_ensembl] = 0
    rna_target_tensor = sparse_scipy_to_tensor(csr_matrix(rna_test_mat.X.astype(float)))
    predictions_perturb_list = []
    for batch_cell in test_loader:
        batch_cell_idx = batch_cell[0].cpu()
        target_rna = rna_target_tensor.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device)
        cell_rna_orig = XDict({'x_seq': rna_test_original.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device)})
        # Extract initial latent representation `z` before perturbation
        with torch.no_grad():
            z_init = model.cellplm(cell_rna_orig, gene_list)[0]['pred']
        z_optim = z_init.detach().clone()
        z_optim.requires_grad_(True)
        optimizer = torch.optim.AdamW([z_optim], lr)
        # Optimize latent space `z` so that the reconstructed RNA matches the silenced RNA state
        for epoch in range(1):
            optimizer.zero_grad()
            cell_rna_2 = XDict({'z': z_optim, 'batch': batch_test[batch_cell_idx].to(device)})
            rna_ = model(cell_rna_2, True)[1]
            ls = rna_loss(rna_.flatten(), torch.log1p(target_rna.to_dense().flatten()))
            ls.backward()
            optimizer.step()
        # Predict post-perturbation ATAC accessibility using the optimized `z`
        with torch.no_grad():
            cell_rna_final = XDict({'z': z_optim.detach(), 'batch': batch_test[batch_cell_idx].to(device)})
            atac_pred = model(cell_rna_final, True)[0]
            predictions_perturb_list.append(atac_pred.cpu().numpy())
        del ls, rna_, optimizer, z_optim, cell_rna_orig, cell_rna_2, cell_rna_final
        torch.cuda.empty_cache()
    predictions_perturb = np.concatenate(predictions_perturb_list, axis=0)
    return predictions_perturb


def tfidf_seurat(adata: ad.AnnData, scale_factor=10000) -> None:
    r"""
    TF-IDF normalization.
    This function normalizes the ATAC peak counts.

    Parameters
    ----------
    adata : ad.AnnData
        Input AnnData object containing raw ATAC counts in `.X`.

    Returns
    -------
    X_tfidf : scipy.sparse matrix or np.ndarray
        TF-IDF normalized matrix.
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
    
    # 3. TF-IDF Calculation with log1p transformation
    if scipy.sparse.issparse(tf):
        X_tfidf = tf.multiply(idf)
        X_tfidf.data = np.log1p(X_tfidf.data)  # log(1+x)
        if hasattr(X_tfidf, 'tocsr'):
            X_tfidf = X_tfidf.tocsr()
    else:
        X_tfidf = tf * idf
        X_tfidf = np.log1p(X_tfidf)
    return X_tfidf


def top_k_metrics(y_true, y_score, k):
    if len(y_true) == 0: 
        return np.nan, np.nan
        
    # Sort scores in descending order
    sorted_idx = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[sorted_idx]
    
    # Actual top K, limited by the total number of candidates
    actual_k = min(k, len(y_true_sorted))
    if actual_k == 0:
        return 0.0, 0.0
        
    top_k_labels = y_true_sorted[:actual_k]
    
    # Top-K Precision: Proportion of True positives within the top K
    precision_at_k = np.sum(top_k_labels) / actual_k
    
    # Top-K Recall: Proportion of True positives in the top K out of all actual positives
    total_positives = np.sum(y_true)
    if total_positives == 0:
        recall_at_k = np.nan
    else:
        recall_at_k = np.sum(top_k_labels) / total_positives
        
    return precision_at_k, recall_at_k


def calculate_celltype_specific_correlation_with_labels(rna_adata, atac_adata, ct_map, gene_name='LEF1', var_name='feature_name',
                                                       celltype_key='cell_type', ct='RGCs', min_pct=0.05, 
                                                       pval_threshold=0.05, corr_threshold=0.1):
    """
    Calculate Pearson correlation between a specific marker gene and all ATAC peaks within a cell type.
    Generates binary Ground Truth labels based on FDR-adjusted p-values and correlation thresholds.
    
    Returns: correlations, p_values, adjusted_p_values, labels
    """
    common_cells = rna_adata.obs_names
    
    if gene_name not in rna_adata.var[var_name].values:
        raise ValueError(f"Gene {gene_name} not found in RNA data")
    
    # Extract 1D gene expression array for the target gene
    gene_expr = pd.Series(
        rna_adata[common_cells, rna_adata.var[var_name]==gene_name].X.toarray().flatten(),
        index=common_cells
    )
    
    # Extract ATAC accessibility matrix for common cells
    atac_data = pd.DataFrame(
        atac_adata[common_cells].X.toarray() if hasattr(atac_adata.X, 'toarray') else atac_adata[common_cells].X,
        index=common_cells,
        columns=atac_adata.var_names
    )
    
    # Initialize result containers
    correlations = pd.DataFrame(index=atac_data.columns, columns=[ct])
    p_values = pd.DataFrame(index=atac_data.columns, columns=[ct])
    labels = pd.DataFrame(0, index=atac_data.columns, columns=[ct])
    
    # Filter cells belonging to the target cell type
    if celltype_key == 'Cluster':
        ct_cells = rna_adata.obs[rna_adata.obs[celltype_key]==ct].index
    else:
        ct_cells = rna_adata.obs[rna_adata.obs[celltype_key].isin(ct_map[ct])].index
    ct_cells = [c for c in ct_cells if c in common_cells]
    print(f"cell type: {ct}, cell num: {len(ct_cells)}")

    # Calculate correlation iteratively for each peak
    for peak in atac_data.columns:
        peak_data = atac_data.loc[:, peak]
        gene_data = gene_expr

        # Filter out extremely sparse peaks to prevent statistical noise
        pct_expressed = (peak_data[ct_cells] > 0).sum() / len(ct_cells)
        if pct_expressed < min_pct:
            correlations.loc[peak, ct] = 0
            p_values.loc[peak, ct] = np.nan
            continue
        
        # Compute Pearson correlation
        if len(np.unique(peak_data)) > 1 and len(np.unique(gene_data)) > 1:
            corr, pval = pearsonr(peak_data, gene_data)
            corr = abs(corr)
            correlations.loc[peak, ct] = corr
            p_values.loc[peak, ct] = pval
        else:
            correlations.loc[peak, ct] = 0
            p_values.loc[peak, ct] = np.nan
    
    # Apply Benjamini-Hochberg FDR correction to p-values
    adjusted_p_values = p_values.copy()
    valid_pvals = p_values[ct].dropna()
    if len(valid_pvals) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            valid_pvals, alpha=pval_threshold, method='fdr_bh'
        )
        adjusted_p_values.loc[valid_pvals.index, ct] = pvals_corrected
        
        # Binarize labels based on corrected p-value and correlation thresholds
        for peak in valid_pvals.index:
            corr_val = correlations.loc[peak, ct]
            adj_pval = pvals_corrected[valid_pvals.index.get_loc(peak)]
            # print(f"adj_pval: {adj_pval}, corr_val: {abs(corr_val)}")
            if adj_pval < pval_threshold and abs(corr_val) > corr_threshold:
                labels.loc[peak, ct] = 1
            else:
                labels.loc[peak, ct] = 0
                correlations.loc[peak, ct] = 0
    
    return correlations, p_values, adjusted_p_values, labels