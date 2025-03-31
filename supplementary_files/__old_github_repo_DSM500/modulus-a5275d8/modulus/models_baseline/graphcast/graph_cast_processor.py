# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from torch import Tensor
from dgl import DGLGraph
from typing import Tuple

from modulus.models_baseline.gnn_layers.mesh_blocks import MeshEdgeBlock, MeshNodeBlock

class GraphCastProcessor(nn.Module):
    """
    Processor block used in GraphCast operating on a latent space
    represented by hierarchy of icosahedral meshes.
    """
    def __init__(
        self,
        processor_layers: int,
        nodes_input_dim: int,
        edges_input_dim: int,
        aggregation_op: str,
        forwarded_args: dict
    ):
        super().__init__()

        edge_block_args = {
            "nodes_input_dim": nodes_input_dim,
            "edges_input_dim": edges_input_dim,
            "output_dim": edges_input_dim,
            "forwarded_args": forwarded_args
        }

        node_block_args = {
            "nodes_input_dim": nodes_input_dim,
            "edges_input_dim": edges_input_dim,
            "output_dim": nodes_input_dim,
            "aggregation_op": aggregation_op,
            "forwarded_args": forwarded_args
        }

        layers = list()

        for _ in range(processor_layers):
            layers.append(MeshEdgeBlock(**edge_block_args))
            layers.append(MeshNodeBlock(**node_block_args))

        self.processor_layers = nn.ModuleList(layers)

    def forward(
        self,
        graph: DGLGraph,
        edge_feats: Tensor,
        node_feats: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        for module in self.processor_layers:
            edge_feats, node_feats = module(edge_feats, node_feats, graph)
        return edge_feats, node_feats