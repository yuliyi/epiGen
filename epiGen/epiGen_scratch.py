import argparse
import time
import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import random
import os
import scanpy as sc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_recall_curve, matthews_corrcoef, f1_score
from accelerate import Accelerator
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from epiGen_models import load_pretrain, sparse_scipy_to_tensor, XDict, ATACformer

# Initialize Accelerator for distributed training and mixed precision
accelerator = Accelerator()

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
    parser.add_argument('--warmup_epochs', type=int, default=100, help='Number of epochs for stage-1 warmup (decoder only).')
    parser.add_argument('--output_z', action="store_true", default=False, help='Flag to output the latent embeddings (z).')
    parser.add_argument('--mod', type=str, default='training', help='Execution mode: "training" or "testing".')
    args = parser.parse_args()
    return args

def calc_metrics(y_true, y_score, fold):
    """Calculate and print evaluation metrics for ATAC-seq peak prediction."""
    try:
        y_pred = binarization(y_score).flatten()
        y_true = y_true.flatten()
        y_score = y_score.flatten()
        p, r, _ = precision_recall_curve(y_true, y_score)
        aupr = auc(r.astype(np.float32), p.astype(np.float32))
        rr, _ = pearsonr(y_true, y_score)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    except ValueError as e:
        print("ValueError:", str(e))
    else:
        accelerator.print('ATAC Evaluation: Folder:{}|Pearson:{:.4f}|AUPR:{:.4f}|MCC:{:.4f}|F1:{:.4f}'
                          .format(fold, rr, aupr, mcc, f1))

def binarization(imputed):
    """
    Binarize the imputed ATAC-seq matrix based on a dynamically calculated threshold.
    A peak is considered accessible (1) if its value exceeds the 99th percentile of 
    its respective cell AND the overall mean of that peak across all cells.
    """
    return ((imputed.T > np.quantile(imputed,q=0.99,axis=1).T).T & (imputed>imputed.mean(0))).astype(np.int8)

@accelerator.on_local_main_process
def save_model(model, args):
    """Save model weights. Ensures it only runs on the main process in distributed settings."""
    model_state_file = os.path.join(args.save_dir, f'model_fold_{args.fold}')
    if not os.path.exists(model_state_file):
        os.makedirs(model_state_file)
    # Unwrap the model from Accelerator wrapper before saving
    model = accelerator.unwrap_model(model)
    accelerator.save(model.state_dict(), os.path.join(model_state_file, f'{args.chr}_decoder_{args.fold}.pt'))

def early_stopping(model, args, epoch, best_epoch, valid_loss, best_loss, bad_counter):
    """Determine if training should stop based on validation loss improvement."""
    if valid_loss < best_loss:
        best_loss = valid_loss
        bad_counter = 0
        best_epoch = epoch
        save_model(model, args)
    else:
        bad_counter += 1
    return bad_counter, best_loss, best_epoch

def load_model(model, args):
    """Load the best saved model weights for testing or resuming training."""
    accelerator.wait_for_everyone() # Synchronize all processes before loading
    model_path = os.path.join(args.save_dir, f'model_fold_{args.fold}/{args.chr}_decoder_{args.fold}.pt')
    model = accelerator.unwrap_model(model)
    model.load_state_dict(torch.load(model_path))
    return model


def validate(model, peak_mat, valid_loader, rna_mat, batch_list, beta, device):
    """Validation loop to compute metrics on the validation set."""
    model.eval()
    rna_loss = nn.MSELoss()
    atac_loss = nn.BCELoss()
    with torch.no_grad():
        total_loss = torch.tensor(0).float().to(device)
        total_l1 = torch.tensor(0).float().to(device)
        for batch_cell in valid_loader:
            batch_cell_idx = batch_cell[0].cpu()
            
            # Construct input dictionary containing RNA expression and technical batch IDs
            cell_rna = XDict({'x_seq': rna_mat.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device),
                              'batch': batch_list[batch_cell_idx].to(device)})
            outs, rna_, ll = model(cell_rna)
            state = peak_mat[batch_cell_idx].to(device)
            l1 = atac_loss(outs.flatten(), state.flatten())
            l2 = rna_loss(rna_.flatten(), torch.log1p(cell_rna['x_seq'].to_dense().flatten()))
            ls = l1 + l2 + beta * ll
            total_loss += ls
            total_l1 += l1
            del ls, l1, l2, state, cell_rna, outs, rna_
            torch.cuda.empty_cache()
            
        mean_loss = total_loss / len(valid_loader)
        mean_l1 = total_l1 / len(valid_loader)
        
        # Aggregate losses across all GPUs
        all_loss = accelerator.reduce(mean_loss, reduction="mean")
        l1_loss = accelerator.reduce(mean_l1, reduction="mean")
        return all_loss, l1_loss

@accelerator.on_local_main_process
def evaluate(model, peak_mat, test_loader, rna_test, batch_test, device, args=None):
    """Inference loop to generate final predictions on the test dataset and compute metrics."""
    model.eval()
    yy_score = []
    yy_true = []
    yy_z = []
    yy_batch = []
    with torch.no_grad():
        for batch_cell in test_loader:
            batch_cell_idx = batch_cell[0].cpu()
            cell_rna = XDict({'x_seq': rna_test.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device),
                              'batch': batch_test[batch_cell_idx].to(device)})
            outs = model(cell_rna, output_z=args.output_z)
            yy_score.append(outs[0].detach().cpu().numpy())
            yy_true.append(peak_mat[batch_cell_idx])
            
            if args.output_z:
                yy_z.append(outs[3].detach().cpu().numpy())
                yy_batch.append(batch_test[batch_cell_idx])
            del cell_rna, outs
        
        # Concatenate predictions across all batches
        a_preds = np.concatenate(yy_score, axis=0)
        a_targets = np.concatenate(yy_true, axis=0)
        
        if args.output_z:
            a_zs = np.concatenate(yy_z, axis=0)
            a_batchs = np.concatenate(yy_batch, axis=0)
            del yy_z, yy_batch
        del yy_score, yy_true
        
        accelerator.print(f"process {accelerator.process_index}, preds_shape: {a_preds.shape}, trues_shape: {a_targets.shape}")
        
        # Save predictions and targets for downstream analysis
        np.save(args.data_dir + f"/prediction/{args.chr}_{args.fold}_preds.npy", a_preds)
        if args.output_z:
            np.save(args.data_dir + f"/prediction/{args.chr}_{args.fold}_embeds.npy", a_zs)
            np.save(args.data_dir + f"/prediction/{args.chr}_{args.fold}_batchs.npy", a_batchs)
            
        calc_metrics(a_targets, a_preds, args.fold)
        del a_preds, a_targets
        torch.cuda.empty_cache()
        return None


def main(args):
    chr = args.chr
    prefix = args.data_dir
    device = accelerator.device
    
    # ---------------------------------------------------------
    # 1. Data Loading and Preprocessing
    # ---------------------------------------------------------
    atac_list = ad.read_h5ad(prefix + f"/atac_data/atac_{chr}_{args.fold}.h5ad")
    rna_list = ad.read_h5ad(prefix + f"/rna_data/rna_{args.fold}.h5ad")
    train_idx = np.load(prefix + f"/train_idx_{args.fold}.npy")
    test_idx = np.load(prefix + f"/test_idx_{args.fold}.npy")

    # Normalize RNA data (Log-normalization is typically done inside the model/loss)
    sc.pp.normalize_total(rna_list, 1e4)
    peak_count = atac_list.shape[1]
    accelerator.print("atac shape:", atac_list.shape)
    
    # Handle batch IDs
    batch_info = rna_list.obs.batch.tolist()
    batch_int, uniques = pd.factorize(batch_info)
    accelerator.print(f"batch size: {len(uniques)}")
    mapping = dict(zip(uniques, range(len(uniques))))
    accelerator.print(f"batch mapping: {mapping}")
    batch_test = torch.from_numpy(batch_int[test_idx].astype(int))
    batch_train = torch.from_numpy(batch_int[train_idx].astype(int))
    accelerator.print(f"batch train unique: {set(batch_train)}")
    accelerator.print(f"batch test unique: {set(batch_test)}")
    gene_list = rna_list.var_names.tolist()
    
    # ---------------------------------------------------------
    # 2. Model Initialization
    # ---------------------------------------------------------
    PRETRAIN_VERSION = '20230926_85M'
    # Load pretrained Foundation Model and freeze its latent components initially
    cellplm = load_pretrain(PRETRAIN_VERSION, args.save_dir)
    cellplm.latent.eval()
    for param in cellplm.latent.parameters():
        param.requires_grad = False
        
    rna_list = sparse_scipy_to_tensor(rna_list.X.astype(float))
    accelerator.print(f"***********training***********")
    stime = time.time()
    atac_train = atac_list[train_idx].X.toarray()
    atac_test = atac_list[test_idx].X.toarray()
    atac_mat_test = torch.from_numpy(atac_test).float()
    rna_test = rna_list.index_select(0, torch.IntTensor(test_idx).type(torch.long))
    
    # Prepare Dataloaders
    testset = torch.utils.data.TensorDataset(torch.IntTensor(range(len(test_idx))))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    rna_train = rna_list.index_select(0, torch.IntTensor(train_idx).type(torch.long))
    atac_mat_train = torch.from_numpy(atac_train).float()
    cell_num_train = len(train_idx)
    accelerator.print("loading data completed...")

    # Split 5% of training data for validation/early stopping
    train_idx, valid_idx = train_test_split(range(cell_num_train), test_size=0.05, random_state=args.seed)
    trainset = torch.utils.data.TensorDataset(torch.IntTensor(train_idx))
    validset = torch.utils.data.TensorDataset(torch.IntTensor(valid_idx))
    
    model = ATACformer(args, cellplm, gene_list, peak_size=peak_count, num_batches=len(uniques))
    rna_loss = nn.MSELoss()
    atac_loss = nn.BCELoss()
    
    # Stage 1 Optimizer: Only train parameters with requires_grad=True (Decoders)
    # The cellplm params group is prepared here but currently has requires_grad=False
    optimizer = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.lr * 3},
                                   {'params': model.cellplm.parameters(), 'lr':args.lr * 2}])

    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    _valid = torch.utils.data.DataLoader(validset, batch_size=100, num_workers=4)

    # Wrap components with Accelerator for multi-GPU support
    model, optimizer, _train, _valid = accelerator.prepare(
        model, optimizer, _train, _valid
    )
    
    if args.mod == 'testing':
        accelerator.print(f"***********testing***********")
        model = load_model(model, args)
        evaluate(model, atac_mat_test, _test, rna_test, batch_test, device, args)
        return
        
    param_num = sum([param.data.numel() for param in model.parameters()])
    accelerator.print('Parameter number: %.3f M' % ((param_num) / 1e6))
    for name, param in model.named_parameters():
        if param.requires_grad:
            accelerator.print(name)
            
    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    bad_counter = 0
    best_epoch = 0
    best_loss = 10000
    warmup_epochs = args.warmup_epochs
    flag1 = True
    flag2 = True
    flag3 = True
    pbar = tqdm(range(1, args.n_epoch + 1))
    
    for epoch in pbar:
        # Ignore early stopping during the warmup phase
        if bad_counter >= args.stopping_steps and epoch < warmup_epochs:
                continue
                
        # --- STAGE 2: Finetuning (Unfreeze Encoder) ---
        if epoch >= warmup_epochs and flag1:
            model = load_model(model, args)
            
            # Re-initialize the optimizer with a much lower learning rate for the decoder
            optimizer = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.lr * 0.3},
                                   {'params': model.cellplm.parameters(), 'lr':args.lr * 2}])
            model, optimizer = accelerator.prepare(model, optimizer)
            accelerator.print("optimizer.param_groups[0]['lr']:", optimizer.param_groups[0]['lr'])
            accelerator.print("optimizer.param_groups[1]['lr']:", optimizer.param_groups[1]['lr'])
            
            # Unfreeze the foundation model CellPLM for joint fine-tuning
            if hasattr(model, 'cellplm'):
                model.cellplm.latent.train()
                for param in model.cellplm.latent.parameters():
                    param.requires_grad = True
                    accelerator.print("cellplm.latent")
            elif hasattr(model, 'module'):
                model.module.cellplm.latent.train()
                for param in model.module.cellplm.latent.parameters():
                    param.requires_grad = True
                    accelerator.print("module.cellplm.latent")
            else:
                accelerator.print("no attribute of cellplm")
            flag1 = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    accelerator.print(name)
                    
        # --- Learning Rate Decay Milestones ---
        if epoch >= warmup_epochs + 200 and flag2:
            optimizer.param_groups[1]['lr'] = args.lr * 1
            flag2 = False
        if epoch >= warmup_epochs + 300 and flag3:
            optimizer.param_groups[1]['lr'] = args.lr * 0.5
            flag3 = False
            
        pbar.set_description("epoch:{}".format(epoch))
        model.train()
        beta = 0.01
        avg_loss = 0
        fp = True
        
        # Batch Training
        for batch_cell in _train:
            batch_cell_idx = batch_cell[0].cpu()
            optimizer.zero_grad()
            state = atac_mat_train[batch_cell_idx].to(device)
            cell_rna = XDict({'x_seq': rna_train.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device),
                              'batch': batch_train[batch_cell_idx].to(device)})
            outs, rna_, ll = model(cell_rna)
            l1 = atac_loss(outs.flatten(), state.flatten())
            l2 = rna_loss(rna_.flatten(), torch.log1p(cell_rna['x_seq'].to_dense().flatten()))
            if epoch >= warmup_epochs:
                ls = l1 + l2 + beta * ll
            else:
                ls = l1 + l2
            if fp:
                accelerator.print("l1:", l1.item(), ", l2:", l2.item(), ", ll:", ll.item())
                fp = False
            del state, cell_rna, outs, rna_, batch_cell
            accelerator.backward(ls)
            optimizer.step()
            avg_loss += ls.item()
            del ls
        accelerator.print('ATACformer Training: Epoch:{}|loss:{:.4f}'.format(epoch, avg_loss))
        # Validation and Early Stopping Check
        if epoch >= warmup_epochs:
            if epoch == warmup_epochs:
                best_loss = 10000 # Reset best loss for the new fine-tuning stage
            valid_loss, valid_l1 = validate(model, atac_mat_train, _valid, rna_train, batch_train, beta, device)
        else:
            valid_loss, valid_l1 = validate(model, atac_mat_train, _valid, rna_train, batch_train, 0, device)
            
        bad_counter, best_loss, best_epoch = early_stopping(model, args, epoch, best_epoch, valid_loss, best_loss, bad_counter)
        accelerator.print('ATAC Validation: Folder:{}| Best_epoch {}|loss:{:.4f}|l1:{:.4f}'.format(1, best_epoch, valid_loss, valid_l1))
        # Evaluation
        if bad_counter >= args.stopping_steps or epoch == args.n_epoch:
            accelerator.print(f"***********testing***********")
            model = load_model(model, args)
            evaluate(model, atac_mat_test, _test, rna_test, batch_test, device, args)
            break
    accelerator.print("*******************Time {:.1f}s".format((time.time()-stime) / 60))


if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    accelerator.print("gpu_num:", gpu_num)
    args = parse_args()
    accelerator.print(args)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)