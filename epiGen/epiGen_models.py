import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math
import warnings
from CellPLM.model import OmicsFormer
from scipy.sparse import csr_matrix
warnings.filterwarnings("ignore")

def load_pretrain(
        pretrain_prefix: str,
        pretrain_directory: str = './ckpt'):
    """Load the pre-trained foundation model CellPLM."""
    config_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.config.json')
    ckpt_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.best.ckpt')
    
    with open(config_path, "r") as openfile:
        config = json.load(openfile)
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

def sparse_scipy_to_tensor(x: csr_matrix):
    """Convert a SciPy sparse matrix to a PyTorch sparse CSR tensor to save memory."""
    return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, (x.shape[0], x.shape[1])).to_sparse().float().coalesce()

class XDict(dict):
    """Custom dictionary to hold complex batch inputs (e.g., RNA seq matrix and batch IDs)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num = self[list(self.keys())[0]].shape[0]

    def size(self):
        warnings.warn("Deprecated function: Xdict.size().", DeprecationWarning)
        return self._num

def expand_batch_embedding(batch_embedding, num_old_batches, num_new_batches=1):
    # Dynamically expand the batch embedding layer to accommodate new datasets during fine-tuning.
    emb_dim = batch_embedding.weight.shape[1]
    new_total_batches = num_old_batches + num_new_batches

    # Instantiate and immediately move to the target device
    new_emb = nn.Embedding(new_total_batches, emb_dim)

    with torch.no_grad():
        new_emb.weight[:num_old_batches] = batch_embedding.weight.clone()
        old_mean = batch_embedding.weight.mean(dim=0)
        old_std = batch_embedding.weight.std(dim=0) + 1e-6
        for i in range(num_old_batches, new_total_batches):
            new_emb.weight[i] = torch.normal(mean=old_mean, std=old_std)
    
    return new_emb

class DecoderLinear(torch.nn.Module):
    """A building block for the Decoder consisting of Linear -> BatchNorm -> Activation."""
    def __init__(
        self,
        in_features,
        out_features,
        base_activation=torch.nn.SiLU,
        bn_layer=True,
    ):
        super(DecoderLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = base_activation(inplace=True)
        self.bn_layer = bn_layer
        
        if bn_layer:
            self.bn = torch.nn.BatchNorm1d(self.in_features)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Kaiming normal distribution."""
        torch.nn.init.kaiming_normal_(self.base_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        if self.bn_layer:
            x = self.bn(x)
        output = F.linear(self.base_activation(x), self.base_weight)
        return output

class Decoder(torch.nn.Module):
    """Multi-layer perceptron acting as the decoder to map latent representations back to specific omics."""
    def __init__(
        self,
        layers_hidden,
        base_activation=torch.nn.SiLU,
    ):
        super(Decoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        # Dynamically build layers based on the hidden dimension list
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                DecoderLinear(
                    in_features,
                    out_features,
                    base_activation=base_activation
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class ATACformer(nn.Module):
    """
    Main model architecture: Cross-modality framework predicting ATAC-seq accessibility from scRNA-seq profiles.
    """
    def __init__(self, args, cellplm, gene_list, cell_dim=512, peak_size=30003, gene_size=15021, num_batches=222):
        super(ATACformer, self).__init__()
        self.embed_dim = args.embed_dim
        self.cell_dim = cell_dim
        self.peak_size = peak_size
        self.gene_size = gene_size
        self.num_batches = num_batches
        self.gene_list = gene_list
        self.cellplm = cellplm
        self.batch_embed = nn.Embedding(self.num_batches, self.embed_dim)
        # decoder for ATAC prediction
        self.ae = nn.Sequential(Decoder([self.cell_dim + self.embed_dim, self.embed_dim*4, self.embed_dim*2, self.embed_dim, self.peak_size]),
                                nn.Sigmoid())
        # decoder for RNA reconstruction
        self.re = Decoder([self.cell_dim + self.embed_dim, self.cell_dim, self.cell_dim, self.gene_size])

    def forward(self, x_dict, perturb=False, output_z=False):
        # Forward pass: extract representations and compute predictions
        if perturb:
            cell_embeddings = x_dict['z']
        else:
            out_dict = self.cellplm(x_dict, self.gene_list)[0]
            cell_embeddings = out_dict['pred']
            
        # Concatenate cell embeddings with learned batch embeddings
        batch_emb = self.batch_embed(x_dict['batch'])
        # Decode specific omics profiles
        ae_rec = self.ae(torch.cat([cell_embeddings, batch_emb], dim=-1))
        re_rec = self.re(torch.cat([cell_embeddings, batch_emb], dim=-1))

        if perturb:
            return ae_rec, re_rec
        else:
            if output_z:
                return ae_rec, re_rec, out_dict['latent_loss'], cell_embeddings
            else:
                return ae_rec, re_rec, out_dict['latent_loss']