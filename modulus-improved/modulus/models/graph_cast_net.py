# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

import time
import yaml
from pathlib import Path

from modulus.models.utils.activations import get_activation

from modulus.utils.graph_utils import latlon2xyz, deg2rad
from modulus.utils.graph_structure import Graph

from modulus.models.components.embedder import GraphCastEncoderEmbedder, GraphCastDecoderEmbedder
from modulus.models.components.encoder_decoder import MeshGraphEncoder, MeshGraphDecoder
from modulus.models.components.processor import GraphCastProcessor
from modulus.models.components.mlp import MeshGraphMLP, MeshGraphEdgeMLPSum

class GraphCastNet(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.device = device

        self._build_coordinates_data()
        self._build_channels_metadata()
        self._build_graphs()
        self._build_model()

        # Move everthing to the intended type and device
        self = super(GraphCastNet, self).to(dtype=self.dtype(), device=self.device)

    def _build_coordinates_data(self):
        _height = self.cfg.dataset.sample.height
        _width = self.cfg.dataset.sample.width

        # Build latlon coordinates for each pixel on the given image
        self.latitudes = torch.linspace(90, -90, steps=_height)
        self.longitudes = torch.linspace(-180, 180, steps=_width+1)[1:]
        
        # Map from image coordinates (721, 1440) to lat/lon coordinates (721, 1440, 2)
        self.map_grid_to_latlon = torch.stack(torch.meshgrid(self.latitudes, self.longitudes, indexing="ij"), dim=-1)

        # List of lat/lon coordinates (1038240, 2)
        self.list_grid_latlon = self.map_grid_to_latlon.view(-1, 2)

        # List of 3D coordinates (1038240, 3)
        self.list_grid_3D_coords = latlon2xyz(self.list_grid_latlon)

        # Map from image coordinates (721, 1440) to 3D coordinates (721, 1440, 3)
        self.map_grid_3D_coords = self.list_grid_3D_coords.view(_height, _width, 3)

        # Normalized area of the latitude-longitude grid cell, of shape (721, 1440)
        area = torch.abs(torch.cos(deg2rad(self.latitudes)))
        area = area / torch.mean(area)
        self.area = area.unsqueeze(1).expand(-1, _width).to(dtype=self.dtype(), device=self.device)

    def _build_channels_metadata(self):
        with open(Path(self.cfg.dataset.base_path) / "metadata.yaml", "r") as f:
            metadata = yaml.safe_load(f)

        if not self.cfg.toggles.data.include_sst_channel:
            del metadata[20] 
        
        self.metadata = metadata

    def _build_graphs(self):
        _graph = Graph(
            list_grid_3D_coords=self.list_grid_3D_coords,
            mesh_level=self.cfg.model.graph.mesh_level,
            use_multimesh=self.cfg.toggles.graph.use_multimesh
        )

        self.m2m_graph = _graph.create_m2m_graph().to(device=self.device)
        self.g2m_graph = _graph.create_g2m_graph().to(device=self.device)
        self.m2g_graph = _graph.create_m2g_graph().to(device=self.device)

        self.m2m_graph.ndata['spatial_info'] = self.m2m_graph.ndata['spatial_info'].to(dtype=self.dtype(), device=self.device)
        self.g2m_graph.edata["displacement_info"] = self.g2m_graph.edata["displacement_info"].to(dtype=self.dtype(), device=self.device)
        self.m2m_graph.edata['displacement_info'] = self.m2m_graph.edata['displacement_info'].to(dtype=self.dtype(), device=self.device)
        self.m2g_graph.edata["displacement_info"] = self.m2g_graph.edata["displacement_info"].to(dtype=self.dtype(), device=self.device)

    def _build_model(self):
        _hidden_dim = self.cfg.model.hidden_dim
        _hidden_layers = self.cfg.model.hidden_layers
        _activation_fn = get_activation(self.cfg.model.activation_fn)
        _aggregation_op = self.cfg.model.aggregation_op
        _processor_layers = self.cfg.model.processor_layers
        _args = {
            "hidden_dim": _hidden_dim,
            "hidden_layers": _hidden_layers,
            "activation_fn": _activation_fn,
            "final_layer_norm": True,
        }

        # initial feature embedder
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_channels=self.input_channels_count(),
            forwarded_args=_args | {"output_dim": _hidden_dim}
        )
        self.decoder_embedder = GraphCastDecoderEmbedder(
            forwarded_args=_args | {"output_dim": _hidden_dim}
        )

        # grid2mesh encoder
        self.encoder = MeshGraphEncoder(
            src_nodes_input_dim=_hidden_dim,
            src_nodes_output_dim=_hidden_dim,
            dst_nodes_input_dim=_hidden_dim,
            dst_nodes_output_dim=_hidden_dim,
            edges_input_dim=_hidden_dim,
            edges_output_dim=_hidden_dim,
            aggregation_op=_aggregation_op,
            forwarded_args=_args
        )

        # icosahedron processor        
        self.processor_encoder = GraphCastProcessor(
            processor_layers=1,
            nodes_input_dim=_hidden_dim,
            edges_input_dim=_hidden_dim,
            aggregation_op=_aggregation_op,
            forwarded_args=_args
        )
        self.processor = GraphCastProcessor(
            processor_layers=_processor_layers - 2,
            nodes_input_dim=_hidden_dim,
            edges_input_dim=_hidden_dim,
            aggregation_op=_aggregation_op,
            forwarded_args=_args
        )
        self.processor_decoder = GraphCastProcessor(
            processor_layers=1,
            nodes_input_dim=_hidden_dim,
            edges_input_dim=_hidden_dim,
            aggregation_op=_aggregation_op,
            forwarded_args=_args
        )

        # mesh2grid decoder
        self.decoder = MeshGraphDecoder(
            src_nodes_input_dim=_hidden_dim,
            dst_nodes_input_dim=_hidden_dim,
            dst_nodes_output_dim=_hidden_dim,
            edges_input_dim=_hidden_dim,
            edges_output_dim=_hidden_dim,
            aggregation_op=_aggregation_op,
            forwarded_args=_args
        )

        # final MLP
        self.finale = MeshGraphMLP(
            input_dim=_hidden_dim,
            output_dim=self.output_channels_count(),
            **_args | {"final_layer_norm": False}
        )

    def forward(self, grid_input):
        # Prepare input
        grid_input = self._prepare_input(grid_input=grid_input)

        # Encoder embedder
        grid_embedded, m2m_node_embedded, g2m_edge_embedded, m2m_edge_embedded = self.checkpoint_filter(partial(
            self.encoder_embedder,
            grid_input=grid_input,
            m2m_node_input=self.m2m_graph.ndata['spatial_info'],
            g2m_edge_input=self.g2m_graph.edata["displacement_info"],
            m2m_edge_input=self.m2m_graph.edata['displacement_info'],
        ))

        # Encoder
        grid_input_encoded, m2m_node_encoded = self.checkpoint_filter(partial(
            self.encoder,
            g2m_graph=self.g2m_graph,
            grid_embedded=grid_embedded,
            m2m_node_embedded=m2m_node_embedded,
            g2m_edge_embedded=g2m_edge_embedded,
        ))

        # Processor encoder
        m2m_edge_processed, m2m_node_processed = self.checkpoint_filter(partial(
            self.processor_encoder,
            graph=self.m2m_graph,
            edge_feats=m2m_edge_embedded,
            node_feats=m2m_node_encoded,
        ))

        # Processor
        m2m_edge_processed, m2m_node_processed = self.checkpoint_filter(partial(
            self.processor,
            graph=self.m2m_graph,
            edge_feats=m2m_edge_processed,
            node_feats=m2m_node_processed,
        ))

        # Processor decoder
        _, m2m_node_processed = self.checkpoint_filter(partial(
            self.processor_decoder,
            graph=self.m2m_graph,
            edge_feats=m2m_edge_processed,
            node_feats=m2m_node_processed,
        ))

        # Decoder embedder
        m2g_edge_embedded = self.checkpoint_filter(partial(
            self.decoder_embedder,
            m2g_edge_input=self.m2g_graph.edata["displacement_info"]
        ))

        # Decoder (from multimesh to lat/lon)
        grid_node_feats_decoded = self.checkpoint_filter(partial(
            self.decoder,
            m2g_graph=self.m2g_graph,
            m2g_edge_embedded=m2g_edge_embedded,
            m2m_node_processed=m2m_node_processed,
            grid_input_encoded=grid_input_encoded
        ))

        # Finale
        grid_output = self.checkpoint_filter(partial(
            self.finale,
            x=grid_node_feats_decoded
        ))

        # Prepare output
        output = self._prepare_output(grid_output=grid_output)
        
        return output

    def checkpoint_filter(self, partial_function):
        return checkpoint(
            partial_function,
            use_reentrant=False,
            preserve_rng_state=False
        )

    def _prepare_input(self, grid_input):
        """
        Prepare input for processing
        e.g. [31, 721, 1440] -> [31, 1038240] -> [1038240, 31]
        """
        grid_input = grid_input.view(self.input_channels_count(), -1).permute(1, 0)

        return grid_input

    def _prepare_output(self, grid_output):
        """
        Prepare output
        For a 30 channels model inputs, this method receives [1038240, 20] and returns [20, 721, 1440]
        """
        grid_output = grid_output.permute(1, 0)
        grid_output = grid_output.view(
            self.output_channels_count(), 
            self.cfg.dataset.sample.height, 
            self.cfg.dataset.sample.width
        )
        return grid_output

    def input_channels_count(self):
        return self.input_channels_count_dataset() + self.input_channels_count_generated()

    def input_channels_count_dataset(self):
        return self.cfg.dataset.sample.channels - int(not self.cfg.toggles.data.include_sst_channel)

    def input_channels_count_generated(self):
        # !!! The following quantities need to be kept aligned with the dataloader, who actually generate the data !!!
        return \
            int(self.cfg.toggles.model.include_static_data) * 2 + \
            int(self.cfg.toggles.model.include_spatial_info) * 3 + \
            int(self.cfg.toggles.model.include_temporal_info) * 4 + \
            int(self.cfg.toggles.model.include_solar_radiation) * 1

    def output_channels_count(self):
        return self.input_channels_count_dataset()

    def dtype(self):
        return torch.bfloat16 if self.cfg.model.dtype == "bfloat16" else torch.float32
