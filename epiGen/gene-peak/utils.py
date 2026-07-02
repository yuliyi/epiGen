import argparse
import os
import numpy as np
from scipy.sparse import csr_matrix
from pyranges import read_gtf
from epiGen_models import expand_batch_embedding, sparse_scipy_to_tensor, XDict


def parse_args():
    """Parse command line arguments for hyperparameters and paths."""
    parser = argparse.ArgumentParser(description="Run epiGen.")
    parser.add_argument('--data_dir', default='/path',
                        help='Input data path.')
    parser.add_argument('--save_dir', default='/path/model',
                        help='Directory to save the trained model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Dimension of the batch embedding.')
    parser.add_argument('--n_epoch', type=int, default=500, help='Total number of training epochs.')
    parser.add_argument('--stopping_steps', type=int, default=100,
                        help='Patience (number of epochs) for early stopping.')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training.')
    parser.add_argument('--chr', type=str, default='chr10', help='Specific chromosome to process.')
    parser.add_argument('--fold', type=str, default='holdout', help='Name of the test dataset or cross-validation dataset.')
    parser.add_argument('--ct', type=str, default='T', help='cell type (e.g. T/B/NK)')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='Number of epochs for stage-1 warmup (decoder only).')
    parser.add_argument('--output_z', action="store_true", default=False, help='Flag to output the latent embeddings (z).')
    parser.add_argument('--mod', type=str, default='training', help='Execution mode: "training" or "testing".')
    args = parser.parse_args()
    return args


def load_model(model, args, chr):
    """
    Load the pre-trained ATACformer decoder weights for a specific chromosome.
    """
    model_path = os.path.join(args.save_dir, f'model/model_fold_pretrain/{chr}_decoder_pretrain.pt')
    model.load_state_dict(torch.load(model_path))
    return model


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

    
import torch

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
    permutation: boolean
        is permutation.
    Returns:
    --------
    predictions_perturb : np.ndarray
        The concatenated ATAC peak predictions after in silico perturbation.
    """
    
    rna_loss = torch.nn.MSELoss()
    # Set the expression of the target gene to 0 to simulate knockout
    if permutation:
        group_col = rna_test_mat.obs['ct_a'].values
        for group_name in np.unique(group_col):
            group_mask = group_col == group_name
            group_indices = np.where(group_mask)[0]
            if len(group_indices) > 0:
                group_data = rna_test_mat[group_indices, gene_ensembl].X.toarray().copy()
                shuffled_data = np.random.permutation(group_data)
                rna_test_mat[group_indices, gene_ensembl] = shuffled_data
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