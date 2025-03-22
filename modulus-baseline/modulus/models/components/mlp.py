# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformer_engine import pytorch as te
from torch.utils.checkpoint import checkpoint
import functools

from .utils import sum_efeat

class MeshGraphMLP(nn.Module):
    """
    MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        activation_fn: nn.Module,
        final_layer_norm: bool,
    ):
        super().__init__()
        MeshGraphMLP.do_checkpoint = False

        additional_hidden_layers = hidden_layers - 1

        layers = list()
        # Entry layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)
        # Hidden layers
        for _ in range(additional_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Norm layer
        if final_layer_norm:
            layers.append(te.LayerNorm(output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if MeshGraphMLP.do_checkpoint:
            return checkpoint(
                self.model,
                x,
                use_reentrant=False,
                preserve_rng_state=False
            )
        else:
            return self.model(x)

class MeshGraphEdgeMLPSum(nn.Module):
    """
    MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes.
    
    It transform edge features - which originally are intended to be a concatenation of previous edge features,
    and the node features of the corresponding source and destinationn nodes - by transorming these three
    features individually through separate linear transformations and then sums them for each edge accordingly. 
    
    The result of this is transformed through the remaining linear layers and activation or norm functions.
    """

    def __init__(
        self,
        edges_input_dim: int,
        src_nodes_input_dim: int,
        dst_nodes_input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module,
        final_layer_norm: bool
    ):
        super().__init__()
        MeshGraphEdgeMLPSum.do_checkpoint = False

        self.edges_input_dim = edges_input_dim
        self.src_nodes_input_dim = src_nodes_input_dim
        self.dst_nodes_input_dim = dst_nodes_input_dim

        _temp = nn.Linear(
            in_features=(
                self.edges_input_dim +
                self.src_nodes_input_dim +
                self.dst_nodes_input_dim
            ),
            out_features=hidden_dim
        )
        _splits = _temp.weight.split([
            self.edges_input_dim,
            self.src_nodes_input_dim,
            self.dst_nodes_input_dim
        ], dim=1)
        self.edges_linear_weights = nn.Parameter(_splits[0])
        self.src_nodes_linear_weights = nn.Parameter(_splits[1])
        self.dst_nodes_linear_weights = nn.Parameter(_splits[2])
        self.bias = _temp.bias

        additional_hidden_layers = hidden_layers - 1

        layers = list()
        # Entry layer
        layers.append(activation_fn)
        # Hidden layers
        for _ in range(additional_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Norm layer
        if final_layer_norm:
            layers.append(te.LayerNorm(output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, edge_feats, node_feats, graph):
        """
        Forward pass of the truncated MLP.
        This uses separate linear layers without bias.
        Bias is added to one MLP, as we sum afterwards. This adds the bias to the total sum, too.
        Having it in one F.linear should allow a fusion of the bias addition while avoiding adding the bias to the "edge-level" result.
        """
        if isinstance(node_feats, Tensor):
            src_feats, dst_feats = node_feats, node_feats
        else:
            src_feats, dst_feats = node_feats

        mlp_edge = F.linear(edge_feats, self.edges_linear_weights, None)
        mlp_src = F.linear(src_feats, self.src_nodes_linear_weights, None)
        mlp_dst = F.linear(dst_feats, self.dst_nodes_linear_weights, self.bias)

        mlp_sum = sum_efeat(
            edge_feats=mlp_edge,
            node_feats=(mlp_src, mlp_dst),
            graph=graph
        )

        if MeshGraphEdgeMLPSum.do_checkpoint:
            return checkpoint(
                self.model,
                mlp_sum,
                use_reentrant=False,
                preserve_rng_state=False
            )
        else:
            return self.model(mlp_sum)

