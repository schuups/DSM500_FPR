# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from torch import Tensor
from dgl import DGLGraph
from typing import Tuple

from .utils import aggregate_and_concat
from .mlp import MeshGraphMLP, MeshGraphEdgeMLPSum

class MeshEdgeBlock(nn.Module):
    """
    Edge block used e.g. in GraphCast or MeshGraphNet
    operating on a latent space represented by a mesh.
    """
    def __init__(
        self,
        nodes_input_dim: int,
        edges_input_dim: int,
        output_dim: int,
        forwarded_args: dict
    ):
        super().__init__()

        self.edge_mlp = MeshGraphEdgeMLPSum(
            edges_input_dim=edges_input_dim,
            src_nodes_input_dim=nodes_input_dim,
            dst_nodes_input_dim=nodes_input_dim,
            output_dim=output_dim,
            **forwarded_args
        )

    def forward(
        self,
        edge_feats: Tensor,
        node_feats: Tensor,
        graph: DGLGraph,
    ) -> Tuple[Tensor, Tensor]:
        edge_feats_new = self.edge_mlp(
            edge_feats=edge_feats, 
            node_feats=node_feats,
            graph=graph
        ) + edge_feats
        return edge_feats_new, node_feats

class MeshNodeBlock(nn.Module):
    """
    Node block used e.g. in GraphCast or MeshGraphNet
    operating on a latent space represented by a mesh.
    """
    def __init__(
        self,
        nodes_input_dim: int,
        edges_input_dim: int,
        output_dim: int,
        aggregation_op: str,
        forwarded_args: dict
    ):
        super().__init__()

        self.aggregation_op = aggregation_op

        self.node_mlp = MeshGraphMLP(
            input_dim=nodes_input_dim + edges_input_dim,
            output_dim=output_dim,
            **forwarded_args
        )
    
    def forward(
        self,
        edge_feats: Tensor,
        node_feats: Tensor,
        graph: DGLGraph,
    ) -> Tuple[Tensor, Tensor]:
        aggregated_feats = aggregate_and_concat(
            edge_feats=edge_feats, 
            node_feats=node_feats, 
            aggregation_op=self.aggregation_op,
            graph=graph, 
        )
        node_features_new = self.node_mlp(aggregated_feats) + node_feats
        # update node features + residual connection
        return edge_feats, node_features_new
