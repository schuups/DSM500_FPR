# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from .mlp import MeshGraphMLP

class GraphCastEncoderEmbedder(nn.Module):
    """
    Feature embedder for:
    - grid input (from dataloader)
    - m2m node spatial info
    - g2m edge displacement info
    - m2m edge displacement info
    """
    def __init__(
        self,
        input_channels,
        forwarded_args
    ):
        super().__init__()

        self.grid_input_mlp = MeshGraphMLP(input_dim=input_channels, **forwarded_args)
        self.m2m_node_mlp   = MeshGraphMLP(input_dim=3, **forwarded_args)
        self.g2m_edge_mlp   = MeshGraphMLP(input_dim=4, **forwarded_args)
        self.m2m_edge_mlp   = MeshGraphMLP(input_dim=4, **forwarded_args)

    def forward(
        self, 
        grid_input, 
        m2m_node_input, 
        g2m_edge_input, 
        m2m_edge_input
    ):
        grid_embedded = self.grid_input_mlp(grid_input)
        m2m_node_embedded = self.m2m_node_mlp(m2m_node_input)
        g2m_edge_embedded = self.g2m_edge_mlp(g2m_edge_input)
        m2m_edge_embedded = self.m2m_edge_mlp(m2m_edge_input)

        return grid_embedded, m2m_node_embedded, g2m_edge_embedded, m2m_edge_embedded

class GraphCastDecoderEmbedder(nn.Module):
    """Feature embedder for m2g edge features"""
    def __init__(
        self,
        forwarded_args
    ):
        super().__init__()
        
        self.m2g_edge_mlp = MeshGraphMLP(input_dim=4, **forwarded_args)

    def forward(
        self,
        m2g_edge_input
    ):
        m2g_edge_embedded = self.m2g_edge_mlp(m2g_edge_input)
        return m2g_edge_embedded