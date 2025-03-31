# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch import Tensor
from dgl import DGLGraph

from modulus.models_baseline.gnn_layers.mlp import MeshGraphMLP, MeshGraphEdgeMLPSum
from modulus.models_baseline.gnn_layers.utils import aggregate_and_concat

class MeshGraphEncoder(nn.Module):
    """
    Encoder which acts on the bipartite graph connecting 
    the regular grid (e.g. representing the input domain) 
    to a mesh (e.g. representing a latent space).
    """
    def __init__(
        self,
        src_nodes_input_dim: int,
        src_nodes_output_dim: int,
        dst_nodes_input_dim: int,
        dst_nodes_output_dim: int,
        edges_input_dim: int,
        edges_output_dim: int,
        aggregation_op: str,
        forwarded_args: dict,
    ):
        super().__init__()

        self.aggregation_op = aggregation_op
        
        # edge MLP
        self.edge_mlp = MeshGraphEdgeMLPSum(
            edges_input_dim=edges_input_dim,
            src_nodes_input_dim=src_nodes_input_dim,
            dst_nodes_input_dim=dst_nodes_input_dim,
            output_dim=edges_output_dim,
            **forwarded_args
        )

        # src node MLP
        self.src_node_mlp = MeshGraphMLP(
            input_dim=src_nodes_input_dim,
            output_dim=src_nodes_output_dim,
            **forwarded_args
        )

        # dst node MLP
        self.dst_node_mlp = MeshGraphMLP(
            input_dim=dst_nodes_input_dim + edges_output_dim,
            output_dim=dst_nodes_output_dim,
            **forwarded_args
        )

    def forward(
        self,
        g2m_graph: DGLGraph,
        grid_embedded: Tensor,
        m2m_node_embedded: Tensor,
        g2m_edge_embedded: Tensor,
    ):
        # Update edge features by concatenating:
        # - existing edge features
        # - input data (from the grid)
        # - node features of the m2m
        edge_feats = self.edge_mlp(
            edge_feats=g2m_edge_embedded,
            node_feats=(
                grid_embedded,
                m2m_node_embedded
            ),
            graph=g2m_graph
        )

        # aggregate messages (edge features) to obtain updated node features
        aggregated_feats = aggregate_and_concat(
            edge_feats=edge_feats,
            node_feats=m2m_node_embedded,
            aggregation_op=self.aggregation_op,
            graph=g2m_graph,
        )
        
        # update src, dst node features + residual connections
        m2m_node_encoded = m2m_node_embedded + self.dst_node_mlp(aggregated_feats)
        grid_input_encoded = grid_embedded + self.src_node_mlp(grid_embedded)
        return grid_input_encoded, m2m_node_encoded

class MeshGraphDecoder(nn.Module):
    def __init__(
        self,
        src_nodes_input_dim: int,
        dst_nodes_input_dim: int,
        dst_nodes_output_dim: int,
        edges_input_dim: int,
        edges_output_dim: int,
        aggregation_op: str,
        forwarded_args: dict,
    ):
        super().__init__()
        self.aggregation_op = aggregation_op
        
        # edge MLP
        self.edge_mlp = MeshGraphEdgeMLPSum(
            edges_input_dim=edges_input_dim,
            src_nodes_input_dim=src_nodes_input_dim,
            dst_nodes_input_dim=dst_nodes_input_dim,
            output_dim=edges_output_dim,
            **forwarded_args
        )

        # dst node MLP
        self.node_mlp = MeshGraphMLP(
            input_dim=dst_nodes_input_dim + edges_output_dim,
            output_dim=dst_nodes_output_dim,
            **forwarded_args
        )

    def forward(
        self,
        m2g_graph: DGLGraph,
        m2g_edge_embedded: Tensor,
        m2m_node_processed: Tensor,
        grid_input_encoded: Tensor,
    ) -> Tensor:
        # update edge features
        edge_feature = self.edge_mlp(
            edge_feats=m2g_edge_embedded,
            node_feats=(m2m_node_processed, grid_input_encoded), 
            graph=m2g_graph
        )
        # aggregate messages (edge features) to obtain updated node features
        aggregated_feats = aggregate_and_concat(
            edge_feats=edge_feature, 
            node_feats=grid_input_encoded,
            aggregation_op=self.aggregation_op,
            graph=m2g_graph, 
        )
        # transformation and residual connection
        grid_node_feats_decoded = self.node_mlp(aggregated_feats) + grid_input_encoded
        return grid_node_feats_decoded