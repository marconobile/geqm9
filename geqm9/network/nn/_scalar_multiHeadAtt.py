from typing import Optional, Union
from e3nn import o3
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from geqtrain.nn import GraphModuleMixin
from geqtrain.nn.allegro._fc import ScalarMLPFunction
from geqtrain.data import AtomicDataDict


class ScalarsMultiHeadSelfAttention(GraphModuleMixin, nn.Module):

    '''
    original_embedding_size = original_dimensionality * num_heads

    '''

    def __init__(
            self,
            num_heads,
            field: str,
            irreps_in = None,
            out_irreps = None
        ):

        super(ScalarsMultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.field = field
        in_irreps = irreps_in[field]
        self.out_irreps = out_irreps or in_irreps

        original_embedding_size = in_irreps.ls.count(0)
        output_size = o3.Irreps(self.out_irreps).dim
        self.original_embedding_size = original_embedding_size
        self.head_dim = original_embedding_size * num_heads

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                field: in_irreps,
                },
            irreps_out={
                field: self.out_irreps,
            }
        )

        self.layer_norm_1 = torch.nn.LayerNorm(self.original_embedding_size)

        # map each input element to 3*head_dim : KQV*self.head_dim
        self.lin_proj = ScalarMLPFunction(
            mlp_input_dimension = self.original_embedding_size,
            mlp_latent_dimensions = [],
            mlp_output_dimension = 3 * self.head_dim,
            mlp_nonlinearity = None,
            weight_norm = True,
            dim = 0,
            has_bias = True,
            use_norm_layer = False,
            zero_init_last_layer_weights = False,
        )

        self.out_proj = ScalarMLPFunction(
            mlp_input_dimension = self.head_dim,
            mlp_latent_dimensions = [],
            mlp_output_dimension = output_size,
            mlp_nonlinearity = None,
            weight_norm = True,
            dim = 0,
            has_bias = True,
            use_norm_layer = True,
            zero_init_last_layer_weights = True,
        )


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        features = data[self.field]
        BATCH = data['batch']
        N = features.size(0)
        H = self.num_heads
        D = self.head_dim

        # apply norm layer

        features = self.layer_norm_1(features)

        # project each el into num_heads * original_embedding_size * 3

        KQV = self.lin_proj(features)

        # split to get K, Q, V

        K, Q, V = KQV.split(D, dim=-1)

        # Reshape for multi-head attention

        Q = rearrange(Q, 'n (h d) -> h n d', h=H)  # [H, N, D]
        K = rearrange(K, 'n (h d) -> h n d', h=H)  # [H, N, D]
        V = rearrange(V, 'n (h d) -> h n d', h=H)  # [H, N, D]

        # Compute attention scores

        attn_scores = torch.einsum('hnd,hmd->hnm', Q, K) / (D ** 0.5)  # [H, N, N]

        # Masked attention

        batch_mask = BATCH.view(1, -1, 1) == BATCH.view(1, 1, -1)  # [1, N, N]
        attn_scores = attn_scores.masked_fill(~batch_mask, float('-inf'))

        # Compute attention weights using softmax

        attn_weights = F.softmax(attn_scores, dim=-1)  # [H, N, N]

        # Apply attention weights to values

        attn_output = torch.einsum('hnm,hmd->hnd', attn_weights, V)  # [H, N, D]

        # Concatenate heads and project

        attn_output = rearrange(attn_output, 'h n d -> n (h d)')  # cat

        # out projection

        out = self.out_proj(attn_output)  # [N, original_embedding_size]

        # residual

        #! todo add other norm layer CHECK IF NEEDED
        #! features = features + out

        data[self.field] = out #features
        return data