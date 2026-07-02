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
from sklearn.metrics import auc, precision_recall_curve
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from epiGen_models import load_pretrain, sparse_scipy_to_tensor, XDict, ATACformer, expand_batch_embedding

# Initialize Accelerator for distributed training
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])


def parse_args():
    # Parse command-line arguments for model configuration, paths, and hyperparameters
    parser = argparse.ArgumentParser(description="Run epiGen.")
    parser.add_argument('--data_dir', nargs='?', default='/path',
                        help='Input data path.')
    parser.add_argument('--save_dir', nargs='?', default='/path/model',
                        help='save_model.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.001,
                        metavar='FLOAT', help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=64,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--n_epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=100,
                        help='Number of epoch for early stopping')
    parser.add_argument('--batch_size', type=int, default=128,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--chr', type=str, default='chr10', help='chromosome')
    parser.add_argument('--fold', type=str, default='holdout', help='test dataset')
    parser.add_argument('--ff_ratio', type=float, default=1.0, help='finetune sample ratio')
    parser.add_argument('--new_ratio', type=float, default=0.8, help='new data ratio in batch')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='warmup stage1 epochs')
    parser.add_argument('--output_z', action="store_true", default=False, help='output z')
    parser.add_argument('--mod', type=str, default='finetune', help='finetune or nofinetune')

    args = parser.parse_args()
    return args


def calc_p(y_true, y_score, fold):
    # Calculate Pearson correlation and Area Under Precision-Recall Curve (AUPR) for ATAC peak evaluation
    try:
        y_true = y_true.flatten()
        y_score = y_score.flatten()
        p, r, _ = precision_recall_curve(y_true, y_score)
        aupr = auc(r.astype(np.float32), p.astype(np.float32))
        rr, _ = pearsonr(y_true, y_score)
    except ValueError as e:
        print("ValueError:", str(e))
    else:
        accelerator.print('ATAC Evaluation: Folder:{}|Pearson:{:.4f}|AUPR:{:.4f}'.format(fold, rr, aupr))


@accelerator.on_local_main_process
def save_model(model, args):
    # Save the model state dictionary on the main process
    model_state_file = os.path.join(args.save_dir, f'model_fold_{args.fold}_{args.mod}')
    model = accelerator.unwrap_model(model)
    accelerator.save(model.state_dict(), os.path.join(model_state_file, f'{args.chr}_decoder_{args.fold}_finetune_{args.ff_ratio}_{args.new_ratio}.pt'))


def early_stopping(model, args, epoch, best_epoch, valid_loss, best_loss, bad_counter):
    # Implement early stopping mechanism to prevent over-fitting
    if valid_loss < best_loss:
        best_loss = valid_loss
        bad_counter = 0
        best_epoch = epoch
        save_model(model, args)
    else:
        bad_counter += 1
    return bad_counter, best_loss, best_epoch


def load_model(model, args, is_pretrain=True):
    # Load model weights, either from a generic pretrain checkpoint or a specific fold's checkpoint
    accelerator.wait_for_everyone()
    if is_pretrain:
        model_path = os.path.join(args.save_dir, f'model_fold_pretrain/{args.chr}_decoder_pretrain.pt')
    else:
        model_path = os.path.join(args.save_dir, f'model_fold_{args.fold}_{args.mod}/{args.chr}_decoder_{args.fold}_{args.mod}_{args.ff_ratio}_{args.new_ratio}.pt')
    model = accelerator.unwrap_model(model)
    model.load_state_dict(torch.load(model_path))

    return model


def validate_mixture_batch(model, peak_mat, valid_loader, rna_mat, batch_list, beta, device, is_old=False):
    model.eval()
    rna_loss = nn.MSELoss()
    atac_loss = nn.BCELoss()
    with torch.no_grad():
        total_loss = torch.tensor(0).float().to(device)
        total_l1 = torch.tensor(0).float().to(device)
        # pbar = tqdm(valid_loader)
        for batch_cell in valid_loader:
            batch_cell_idx = batch_cell[0].cpu()
            # Prepare input dictionary with RNA sequence and batch ID
            cell_rna = XDict({'x_seq': sparse_scipy_to_tensor(rna_mat[batch_cell_idx.numpy()].X.astype(float)).to(device),
                              'batch': batch_list[batch_cell_idx].to(device)})
            outs, rna_, ll = model(cell_rna)
            # Retrieve ground truth peak data (handle differences between sparse and dense formats)
            if is_old:
                state = torch.from_numpy(peak_mat[batch_cell_idx.numpy()].X.toarray()).float().to(device)
            else:
                state = peak_mat[batch_cell_idx].to(device)
            # Calculate individual loss components
            l1 = atac_loss(outs.flatten(), state.flatten())
            l2 = rna_loss(rna_.flatten(), torch.log1p(cell_rna['x_seq'].to_dense().flatten()))
            ls = l1 + l2 + beta * ll
            total_loss += ls
            total_l1 += l1
            del ls, l1, l2, state, cell_rna, outs, rna_
            # torch.cuda.empty_cache()
        mean_loss = total_loss / len(valid_loader)
        mean_l1 = total_l1 / len(valid_loader)
        # Aggregate losses across all GPUs
        all_loss = accelerator.reduce(mean_loss, reduction="mean")
        l1_loss = accelerator.reduce(mean_l1, reduction="mean")
        return all_loss, l1_loss


@accelerator.on_local_main_process
def evaluate(model, peak_mat, test_loader, rna_test, batch_test, device, args=None):
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
            # Optionally collect latent embeddings (Z) for downstream analysis
            if args.output_z:
                yy_z.append(outs[3].detach().cpu().numpy())
                yy_batch.append(batch_test[batch_cell_idx])
            del cell_rna, outs
            # torch.cuda.empty_cache()
        # Concatenate batch predictions into full arrays
        a_preds = np.concatenate(yy_score, axis=0)
        a_targets = np.concatenate(yy_true, axis=0)
        if args.output_z:
            a_zs = np.concatenate(yy_z, axis=0)
            a_batchs = np.concatenate(yy_batch, axis=0)
            del yy_z, yy_batch
        del yy_score, yy_true
        # torch.cuda.empty_cache()
        accelerator.print(f"process {accelerator.process_index}, preds_shape: {a_preds.shape}, trues_shape: {a_targets.shape}")
        # Save predictions to disk based on the current mode (finetune or non-fine-tune)
        if args.mod == 'finetune':
            np.save(args.data_dir + f"/prediction/{args.chr}_{args.fold}_preds_{args.mod}_{args.ff_ratio}.npy", a_preds)
        else:
            np.save(args.data_dir + f"/prediction/{args.chr}_{args.fold}_preds_{args.mod}.npy", a_preds)
        if args.output_z:
            np.save(args.data_dir + f"/prediction/{args.chr}_{args.fold}_embeds.npy", a_zs)
            np.save(args.data_dir + f"/prediction/{args.chr}_{args.fold}_batchs.npy", a_batchs)
        # Calculate evaluation metrics
        calc_p(a_targets, a_preds, f"{args.fold}_{args.mod}")
        del a_preds, a_targets
        # torch.cuda.empty_cache()
        return None


def nofinetune_main(args):
    chr = args.chr
    prefix = args.data_dir
    device = accelerator.device
    # Load test indices and prepare datasets
    test_idx = np.load(prefix + f"/test_idx_{args.fold}.npy")
    rna_test = ad.read_h5ad(prefix + f"/rna_data/rna_{args.fold}.h5ad")
    batch_info = rna_test.obs.batch.tolist()
    batch_test, uniques_test = pd.factorize(batch_info)
    # Offset batch indices to avoid overlapping with pretrained batches
    batch_test = torch.from_numpy(np.array(list(map(lambda x: x + 222, batch_test))).astype(int))
    accelerator.print(f"batch size: {len(uniques_test)}")
    mapping = dict(zip(uniques_test, range(222, 222+len(uniques_test))))
    accelerator.print(f"batch mapping: {mapping}")
    rna_test = rna_test[test_idx].copy()
    atac_test = ad.read_h5ad(prefix + f"/atac_data/atac_{chr}_independence_{args.fold}.h5ad")[test_idx].X.toarray()
    batch_test = batch_test[test_idx]
    sc.pp.normalize_total(rna_test, 1e4)
    peak_count = atac_test.shape[1]
    atac_mat_test = torch.from_numpy(atac_test).float()
    gene_list = rna_test.var_names.tolist()
    # Load Foundation Model
    PRETRAIN_VERSION = '20230926_85M'
    cellplm = load_pretrain(PRETRAIN_VERSION, args.save_dir)
    rna_test = sparse_scipy_to_tensor(rna_test.X.astype(float))
    stime = time.time()

    testset = torch.utils.data.TensorDataset(torch.IntTensor(range(len(atac_test))))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    accelerator.print("loading data completed...")
    
    model = ATACformer(args, cellplm, gene_list, peak_size=peak_count)
    rna_loss = nn.MSELoss()
    model = load_model(model, args)
    # train only the batch embeddings to map the new dataset space
    accelerator.print(f"***********traing new batch embedding***********")
    _, ff_idx = train_test_split(range(len(test_idx)), test_size=0.2, random_state=args.seed)
    ffset = torch.utils.data.TensorDataset(torch.IntTensor(ff_idx))
    _ff = torch.utils.data.DataLoader(ffset, batch_size=args.batch_size, num_workers=4)
    # Expand embedding layer and freeze other parameters
    model.batch_embed = expand_batch_embedding(model.batch_embed, num_old_batches=222, num_new_batches=len(uniques_test))
    for param in model.parameters():
        param.requires_grad = False
    for param in model.batch_embed.parameters():
        param.requires_grad = True
    optimizer_batch = torch.optim.AdamW(params=model.batch_embed.parameters(), lr=0.0001)
    model, optimizer_batch, _ff = accelerator.prepare(model, optimizer_batch, _ff)
    model.eval()
    # Train batch embeddings
    for epoch_batch in range(50):
        fp = True
        for batch_cell in _ff:
            optimizer_batch.zero_grad()
            batch_cell_idx = batch_cell[0].cpu()
            cell_rna = XDict({'x_seq': rna_test.index_select(0, torch.IntTensor(batch_cell_idx).type(torch.long)).to(device),
                            'batch': batch_test[batch_cell_idx].to(device)})
            _, rna_, _ = model(cell_rna)
            ls = rna_loss(rna_.flatten(), torch.log1p(cell_rna['x_seq'].to_dense().flatten()))
            accelerator.backward(ls)
            if fp:
                accelerator.print("ls:", ls.item())
                fp = False
            optimizer_batch.step()
    accelerator.print(f"***********testing***********")
    evaluate(model, atac_mat_test, _test, rna_test, batch_test, device, args)
    accelerator.print("*******************Time {:.1f}s".format((time.time()-stime) / 60))


def finetune_mixture_batch_sample(args):
    """
    Full fine-tuning process for ATACformer.
    Incorporates the new target dataset and optionally the original pretraining dataset 
    to mitigate catastrophic forgetting, dynamically controlled by `args.new_ratio`.
    """
    chr = args.chr
    prefix = args.data_dir
    device = accelerator.device
    
    # Robustness switch: Determine whether to mix old pretrain data 
    # If new_ratio is 1.0, we exclusively use the new target dataset.
    use_mixture = args.new_ratio < 1.0

    if use_mixture:
        # 1. Load Pretraining Data (Only executed if mixing is required)
        atac_pretrain = ad.read_h5ad(prefix + f"/atac_data/atac_{chr}_pretrain_train.h5ad", backed='r')
        rna_pretrain = ad.read_h5ad(prefix + f"/rna_data/rna_cellplm_pretrain_train_normalize.h5ad", backed='r')
        batch_info_old = rna_pretrain.obs.batch.tolist()
        batch_pretrain, uniques_old = pd.factorize(batch_info_old)
        batch_pretrain = torch.from_numpy(batch_pretrain.astype(int))
        
        accelerator.print(f"pretrain batch size: {len(uniques_old)}")
        mapping_old = dict(zip(uniques_old, range(len(uniques_old))))
        accelerator.print(f"pretrain batch mapping: {mapping_old}")

    # 2. Load New (Fine-tuning Target) Data
    atac_list = ad.read_h5ad(prefix + f"/atac_data/atac_{chr}_{args.fold}.h5ad")
    rna_list = ad.read_h5ad(prefix + f"/rna_data/rna_{args.fold}.h5ad")
    train_idx = np.load(prefix + f"/train_idx_{args.fold}.npy")
    test_idx = np.load(prefix + f"/test_idx_{args.fold}.npy")
    
    batch_info = rna_list.obs.batch.tolist()
    batch_info, uniques = pd.factorize(batch_info)
    
    # Shift batch indices for the new data to be strictly greater than old pretrain batches.
    batch_info = torch.from_numpy(np.array(list(map(lambda x: x + 222, batch_info))).astype(int))
    
    accelerator.print(f"batch size: {len(uniques)}")
    mapping = dict(zip(uniques, range(222, 222+len(uniques))))
    accelerator.print(f"batch mapping: {mapping}")
    
    peak_count = atac_list.shape[1]
    cell_num = len(train_idx)
    
    # Downsample fine-tuning dataset if ff_ratio is specified
    if args.ff_ratio < 1:
        _, finetune_idx = train_test_split(range(cell_num), test_size=args.ff_ratio, random_state=args.seed)
        train_idx = train_idx[finetune_idx]
        
    if use_mixture:
        # Calculate the sampling ratio between old and new data streams
        rat = (1 - args.new_ratio) / args.new_ratio
        old_idx = range(len(rna_pretrain))
        # Ensure the old data stream size is proportional to the new data batch size
        old_train_idx, old_valid_idx = train_test_split(old_idx, test_size=int(0.05*len(train_idx)*rat), random_state=args.seed)
        accelerator.print("pretrain size:", rna_pretrain.shape[0])

    # Generator to loop over a dataloader infinitely (used for the old data stream)
    def get_infinite_iterator(dl):
        while True:
            for batch in dl:
                yield batch

    # Data Preprocessing
    sc.pp.normalize_total(rna_list, 1e4)
    gene_list = rna_list.var_names.tolist()
    PRETRAIN_VERSION = '20230926_85M'
    
    # Initialize base CellPLM model
    cellplm = load_pretrain(PRETRAIN_VERSION, args.save_dir)
    accelerator.print(f"rna shape:{rna_list.shape}, atac shape:{atac_list.shape}")
    accelerator.print(f"***********training***********")
    stime = time.time()

    # Prepare dense ATAC matrices and sparse RNA tensors for the testing set
    atac_train = atac_list[train_idx].X.toarray()
    atac_test = atac_list[test_idx].X.toarray()
    atac_mat_test = torch.from_numpy(atac_test).float()
    rna_test = sparse_scipy_to_tensor(rna_list[test_idx].X.astype(float))
    testset = torch.utils.data.TensorDataset(torch.IntTensor(range(len(test_idx))))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    
    best_epoch = 0
    rna_train = rna_list[train_idx].copy()
    atac_mat_train = torch.from_numpy(atac_train).float()
    cell_num_train = len(atac_train)
    batch_train = batch_info[train_idx]
    batch_test = batch_info[test_idx]
    accelerator.print("loading data completed...")
    
    # Define training and validation splits for the target dataset
    train_idx, valid_idx = train_test_split(range(cell_num_train), test_size=0.05, random_state=args.seed)
    trainset = torch.utils.data.TensorDataset(torch.IntTensor(train_idx))
    validset = torch.utils.data.TensorDataset(torch.IntTensor(valid_idx))
    
    if use_mixture:
        # Define training and validation splits for the old pretrain dataset
        old_trainset = torch.utils.data.TensorDataset(torch.IntTensor(old_train_idx))
        old_validset = torch.utils.data.TensorDataset(torch.IntTensor(old_valid_idx))
    
    # Initialize the complete ATACformer model
    model = ATACformer(args, cellplm, gene_list, peak_size=peak_count)
    atac_loss = nn.BCELoss()
    rna_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.1)
    
    # Initialize dataloaders dynamically based on the mixture flag
    _valid = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, num_workers=4)
    
    if use_mixture:
        # Apportion the total batch size based on new_ratio
        _train = torch.utils.data.DataLoader(trainset, batch_size=int(args.batch_size * args.new_ratio), shuffle=True, num_workers=4)
        _train_old = torch.utils.data.DataLoader(old_trainset, batch_size=int(args.batch_size * (1 - args.new_ratio)), shuffle=True, num_workers=4)
        _valid_old = torch.utils.data.DataLoader(old_validset, batch_size=args.batch_size, num_workers=4)
    else:
        # Fallback to standard full batch size if no mixture is required
        _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load pretrained weights and expand the batch embedding layer to accommodate new dataset batches
    model = load_model(model, args)
    model.batch_embed = expand_batch_embedding(model.batch_embed, num_old_batches=222, num_new_batches=len(uniques))
    
    # Prepare hardware accelerator components dynamically
    if use_mixture:
        model, optimizer, _train, _valid, _train_old, _valid_old = accelerator.prepare(
            model, optimizer, _train, _valid, _train_old, _valid_old
        )
        # Create an infinite stream to continuously sample from the old dataset
        train_old_stream = iter(get_infinite_iterator(_train_old))
    else:
        model, optimizer, _train, _valid = accelerator.prepare(
            model, optimizer, _train, _valid
        )
        
    accelerator.print("pretained model loaded...")
    bad_counter = 0
    beta = 0.01
    best_loss = 10000
    fp = True
    pbar = tqdm(range(1, args.n_epoch + 1))
    
    # ---------------------------------------------------------
    # Core Training Loop
    # ---------------------------------------------------------
    for epoch in pbar:
        pbar.set_description("epoch:{}".format(epoch))
        model.train()
        avg_loss = 0
        fp = True
        
        for batch_cell in _train:
            batch_cell_idx = batch_cell[0].cpu()
            optimizer.zero_grad()
            
            if use_mixture:
                # 1. Fetch indices for the old data stream
                old_batch_cell_idx = next(train_old_stream)[0].cpu()
                # 2. Retrieve corresponding old ATAC matrices
                atac_old = torch.from_numpy(atac_pretrain[old_batch_cell_idx.numpy()].X.toarray()).float()
                # 3. Concatenate new and old inputs into unified batches
                state = torch.vstack([atac_mat_train[batch_cell_idx], atac_old]).to(device)
                rna_seq = sparse_scipy_to_tensor(ad.concat([rna_train[batch_cell_idx.numpy()], rna_pretrain[old_batch_cell_idx.numpy()]], axis=0).X.astype(float))
                b_cat = torch.cat([batch_train[batch_cell_idx], batch_pretrain[old_batch_cell_idx]], axis=0).to(device)
                cell_rna = XDict({'x_seq': rna_seq.to(device), 'batch': b_cat})
            else:
                # Direct assignment for a single target dataset (avoids concatenation overhead)
                state = atac_mat_train[batch_cell_idx].to(device)
                rna_seq = sparse_scipy_to_tensor(rna_train[batch_cell_idx.numpy()].X.astype(float))
                cell_rna = XDict({'x_seq': rna_seq.to(device), 'batch': batch_train[batch_cell_idx].to(device)})

            # Forward pass
            outs, rna_, ll = model(cell_rna)
            
            # Loss computation: ATAC BCE Loss + RNA MSE Loss + Latent KL Divergence
            l1 = atac_loss(outs.flatten(), state.flatten())
            l2 = rna_loss(rna_.flatten(), torch.log1p(cell_rna['x_seq'].to_dense().flatten()))
            ls = l1 + l2 + beta * ll
            
            # Print loss components for the first iteration of each epoch
            if fp:
                accelerator.print("l1:", l1.item(), ", l2:", l2.item(), ", ll:", ll.item())
                fp = False
                
            # Backward pass and optimization
            accelerator.backward(ls)
            optimizer.step()
            avg_loss += ls.item()
            
            # Memory management
            del ls, state, cell_rna, outs, rna_
            
        accelerator.print('ATACformer Training: Epoch:{}|loss:{:.4f}'.format(epoch, avg_loss))
        
        # ---------------------------------------------------------
        # Validation and Early Stopping
        # ---------------------------------------------------------
        valid_loss, valid_l1 = validate_mixture_batch(model, atac_mat_train, _valid, rna_train, batch_train, beta, device)
        
        if use_mixture:
            old_valid_loss, old_valid_l1 = validate_mixture_batch(model, atac_pretrain, _valid_old, rna_pretrain, batch_pretrain, beta, device, is_old=True)
            accelerator.print('ATAC Validation: Folder:{}| Best_epoch {}|loss:{:.4f}|l1:{:.4f}|old_loss:{:.4f}'.format(1, best_epoch, valid_loss, valid_l1, old_valid_loss))
        else:
            accelerator.print('ATAC Validation: Folder:{}| Best_epoch {}|loss:{:.4f}|l1:{:.4f} (No Pretrain Mixture)'.format(1, best_epoch, valid_loss, valid_l1))
            
        bad_counter, best_loss, best_epoch = early_stopping(model, args, epoch, best_epoch, valid_loss, best_loss, bad_counter)
        
        # Terminate training if early stopping patience is reached or max epochs attained
        if bad_counter >= args.stopping_steps or epoch == args.n_epoch:
            accelerator.print(f"***********testing***********")
            model = load_model(model, args, False)
            evaluate(model, atac_mat_test, _test, rna_test, batch_test, device, args)
            break
            
    accelerator.print("*******************Time {:.1f}s".format((time.time()-stime) / 60))


if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    accelerator.print("gpu_num:", gpu_num)
    args = parse_args()
    accelerator.print(args)
    # seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Route to either the non-finetune execution path or the fine-tuning path
    if args.mod == 'nofinetune':
        nofinetune_main(args)
    elif args.mod == 'finetune':
        finetune_mixture_batch_sample(args)
