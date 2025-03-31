# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import functools

# Original stuff
from modulus.models.layers import get_activation
from modulus.models.meta import ModelMetaData
from modulus.utils.graphcast.graph_utils import latlon2xyz
from modulus.models.layers import get_activation
from modulus.utils.graphcast.graph_utils import deg2rad

# New stuff
from modulus.models_baseline.module import Module

from modulus.utils_new.graphcast.graph import Graph
from modulus.models_baseline.gnn_layers.embedder import (
    GraphCastEncoderEmbedder,
    GraphCastDecoderEmbedder
)
from modulus.models_baseline.gnn_layers.encoder_decoder import (
    MeshGraphEncoder,
    MeshGraphDecoder
)
from modulus.models_baseline.graphcast.graph_cast_processor import (
    GraphCastProcessor
)
from modulus.models_baseline.gnn_layers.mlp import (
    MeshGraphMLP,
    MeshGraphEdgeMLPSum
)

@dataclass
class MetaData(ModelMetaData):
    name: str = "GraphCastNetBaseline"
    # The only two differences from the base class
    # TODO: Check if and how the following parameters are used
    amp_gpu: bool = True
    bf16: bool = True

class GraphCastNetBaseline(Module):
    def __init__(
        self,
        sample_height: int,
        sample_width: int,
        sample_channels: int,

        include_static_data: bool,
        include_spatial_info: bool,
        include_temporal_info: bool,
        include_solar_radiation: bool,

        batch_size: int = 1,

        mesh_level: int = 6,
        activation_fn: str = "silu",
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        aggregation_op: str = "sum",
        processor_layers: int = 16,

        cache_enabled: bool = False
    ):
        super().__init__(meta=MetaData())

        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_channels = sample_channels

        self._include_static_data = include_static_data
        self._include_spatial_info = include_spatial_info
        self._include_temporal_info = include_temporal_info
        self._include_solar_radiation = include_solar_radiation

        self.batch_size = batch_size

        self.mesh_level = mesh_level
        self.activation_fn = get_activation(activation_fn)
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.aggregation_op = aggregation_op
        self.processor_layers = processor_layers
        assert processor_layers > 2, f"'processor_layers' must be > 2, gotten {processor_layers}"
        
        self.checkpoint_enabled = False
        self.cache_enabled = cache_enabled

        self._build_coordinates_data()
        self._build_graphs()
        self._build_model()

    def _build_coordinates_data(self):
        # Build latlon coordinates for each pixel on the given image
        self.latitudes = torch.linspace(90, -90, steps=self.sample_height)
        self.longitudes = torch.linspace(-180, 180, steps=self.sample_width+1)[1:]
        
        # Map from image coordinates (721, 1440) to lat/lon coordinates (721, 1440, 2)
        self.map_grid_to_latlon = torch.stack(torch.meshgrid(self.latitudes, self.longitudes, indexing="ij"), dim=-1)

        # List of lat/lon coordinates (1038240, 2)
        self.list_grid_latlon = self.map_grid_to_latlon.view(-1, 2)

        # List of 3D coordinates (1038240, 3)
        self.list_grid_3D_coords = latlon2xyz(self.list_grid_latlon)

        # Map from image coordinates (721, 1440) to 3D coordinates (721, 1440, 3)
        self.map_grid_3D_coords = self.list_grid_3D_coords.view(self.latitudes.shape[0], self.longitudes.shape[0], 3)

        # Normalized area of the latitude-longitude grid cell
        area = torch.abs(torch.cos(deg2rad(self.latitudes)))
        area = area / torch.mean(area)
        self.area = area.unsqueeze(1).expand(-1, self.sample_width)

    def _build_graphs(self):
        self.graph = Graph(
            list_grid_3D_coords=self.list_grid_3D_coords, 
            list_grid_latlon=self.list_grid_latlon, 
            mesh_level=self.mesh_level,
            cache_enabled=self.cache_enabled
        )
        self.m2m_graph = self.graph.create_m2m_graph()
        self.g2m_graph = self.graph.create_g2m_graph()
        self.m2g_graph = self.graph.create_m2g_graph()
        self.m2m_node_input = self.m2m_graph.ndata['spatial_info']
        self.g2m_edge_input = self.g2m_graph.edata["displacement_info"]
        self.m2m_edge_input = self.m2m_graph.edata['displacement_info']
        self.m2g_edge_input = self.m2g_graph.edata["displacement_info"]

    def _build_model(self):
        _args = {
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "activation_fn": self.activation_fn,
            "final_layer_norm": True,
        }

        # initial feature embedder
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_channels=self.input_channels_count,
            forwarded_args=_args | {"output_dim": self.hidden_dim}
        )
        self.decoder_embedder = GraphCastDecoderEmbedder(
            forwarded_args=_args | {"output_dim": self.hidden_dim}
        )

        # grid2mesh encoder
        self.encoder = MeshGraphEncoder(
            src_nodes_input_dim=self.hidden_dim,
            src_nodes_output_dim=self.hidden_dim,
            dst_nodes_input_dim=self.hidden_dim,
            dst_nodes_output_dim=self.hidden_dim,
            edges_input_dim=self.hidden_dim,
            edges_output_dim=self.hidden_dim,
            aggregation_op=self.aggregation_op,
            forwarded_args=_args
        )

        # icosahedron processor        
        self.processor_encoder = GraphCastProcessor(
            processor_layers=1,
            nodes_input_dim=self.hidden_dim,
            edges_input_dim=self.hidden_dim,
            aggregation_op=self.aggregation_op,
            forwarded_args=_args
        )
        self.processor = GraphCastProcessor(
            processor_layers=self.processor_layers - 2,
            nodes_input_dim=self.hidden_dim,
            edges_input_dim=self.hidden_dim,
            aggregation_op=self.aggregation_op,
            forwarded_args=_args
        )
        self.processor_decoder = GraphCastProcessor(
            processor_layers=1,
            nodes_input_dim=self.hidden_dim,
            edges_input_dim=self.hidden_dim,
            aggregation_op=self.aggregation_op,
            forwarded_args=_args
        )

        # mesh2grid decoder
        self.decoder = MeshGraphDecoder(
            src_nodes_input_dim=self.hidden_dim,
            dst_nodes_input_dim=self.hidden_dim,
            dst_nodes_output_dim=self.hidden_dim,
            edges_input_dim=self.hidden_dim,
            edges_output_dim=self.hidden_dim,
            aggregation_op=self.aggregation_op,
            forwarded_args=_args
        )

        # final MLP
        self.finale = MeshGraphMLP(
            input_dim=self.hidden_dim,
            output_dim=self.output_channels_count,
            **_args | {"final_layer_norm": False}
        )

    def forward(self, grid_input):
        # Prepare input
        grid_input = self._prepare_input(grid_input=grid_input)
        m2m_node_input = self.m2m_node_input
        g2m_edge_input = self.g2m_edge_input
        m2m_edge_input = self.m2m_edge_input
        m2g_edge_input = self.m2g_edge_input

        # Encoder embedder
        grid_embedded, m2m_node_embedded, g2m_edge_embedded, m2m_edge_embedded = self.checkpoint_filter(functools.partial(
            self.encoder_embedder,
            grid_input=grid_input,
            m2m_node_input=m2m_node_input,
            g2m_edge_input=g2m_edge_input,
            m2m_edge_input=m2m_edge_input,
        ))
        # grid_embedded, m2m_node_embedded, g2m_edge_embedded, m2m_edge_embedded = self.encoder_embedder(
        #     grid_input=grid_input,
        #     m2m_node_input=m2m_node_input,
        #     g2m_edge_input=g2m_edge_input,
        #     m2m_edge_input=m2m_edge_input,
        # )

        # Encoder
        grid_input_encoded, m2m_node_encoded = self.checkpoint_filter(functools.partial(
            self.encoder,
            g2m_graph=self.g2m_graph,
            grid_embedded=grid_embedded,
            m2m_node_embedded=m2m_node_embedded,
            g2m_edge_embedded=g2m_edge_embedded,
        ))
        # grid_input_encoded, m2m_node_encoded = self.encoder(
        #     g2m_graph=self.g2m_graph,
        #     grid_embedded=grid_embedded,
        #     m2m_node_embedded=m2m_node_embedded,
        #     g2m_edge_embedded=g2m_edge_embedded,
        # )

        # Processor encoder
        m2m_edge_processed, m2m_node_processed = self.checkpoint_filter(functools.partial(
            self.processor_encoder,
            graph=self.m2m_graph,
            edge_feats=m2m_edge_embedded,
            node_feats=m2m_node_encoded,
        ))
        # m2m_edge_processed, m2m_node_processed = self.processor_encoder(
        #     graph=self.m2m_graph,
        #     edge_feats=m2m_edge_embedded,
        #     node_feats=m2m_node_encoded,
        # )

        # Processor
        m2m_edge_processed, m2m_node_processed = self.checkpoint_filter(functools.partial(
            self.processor,
            graph=self.m2m_graph,
            edge_feats=m2m_edge_processed,
            node_feats=m2m_node_processed,
        ))
        # m2m_edge_processed, m2m_node_processed = self.processor(
        #     graph=self.m2m_graph,
        #     edge_feats=m2m_edge_processed,
        #     node_feats=m2m_node_processed,
        # )

        # Processor decoder
        _, m2m_node_processed = self.checkpoint_filter(functools.partial(
            self.processor_decoder,
            graph=self.m2m_graph,
            edge_feats=m2m_edge_processed,
            node_feats=m2m_node_processed,
        ))
        # _, m2m_node_processed = self.processor_decoder(
        #     graph=self.m2m_graph,
        #     edge_feats=m2m_edge_processed,
        #     node_feats=m2m_node_processed,
        # )

        # Decoder embedder
        m2g_edge_embedded = self.checkpoint_filter(functools.partial(
            self.decoder_embedder,
            m2g_edge_input=m2g_edge_input
        ))
        # m2g_edge_embedded = self.decoder_embedder(
        #     m2g_edge_input=m2g_edge_input
        # )

        # Decoder (from multimesh to lat/lon)
        grid_node_feats_decoded = self.checkpoint_filter(functools.partial(
            self.decoder,
            m2g_graph=self.m2g_graph,
            m2g_edge_embedded=m2g_edge_embedded,
            m2m_node_processed=m2m_node_processed,
            grid_input_encoded=grid_input_encoded
        ))
        # grid_node_feats_decoded = checkpoint(self.decoder(
        #     m2g_graph=self.m2g_graph,
        #     m2g_edge_embedded=m2g_edge_embedded,
        #     m2m_node_processed=m2m_node_processed,
        #     grid_input_encoded=grid_input_encoded
        # ))

        # Finale
        grid_output = self.checkpoint_filter(functools.partial(
            self.finale,
            x=grid_node_feats_decoded
        ))
        #grid_output = self.finale(grid_node_feats_decoded)

        # Prepare output
        output = self._prepare_output(grid_output=grid_output)
        
        return output

    def update_checkpoint_for_rollout_step(self, rollout_steps):
        if rollout_steps == 2:
            MeshGraphMLP.do_checkpoint = True
            MeshGraphEdgeMLPSum.do_checkpoint = True
        elif rollout_steps >= 3:
            self.checkpoint_enabled = True

    def checkpoint_filter(self, partial_function):
        if self.checkpoint_enabled:
            return checkpoint(
                partial_function,
                use_reentrant=False,
                preserve_rng_state=False
            )
        else:
            return partial_function()

    def _prepare_input(self, grid_input):
        """
        Prepare input, i.e. [31, 721, 1440] -> [1038240, 31]
                            [ C,   H,    W]
        """
        assert grid_input.shape == (self.input_channels_count, self.sample_height, self.sample_width), \
            f"Expected shape ({self.input_channels_count}, {self.sample_height}, {self.sample_width}), got {grid_input.shape}"

        grid_input = grid_input.view(self.input_channels_count, -1).permute(1, 0)
        return grid_input

    def _prepare_output(self, grid_output):
        """
        Prepare output, i.e. [1038240, 31] -> [21, 721, 1440]
                                              [C,    H,    W] where and S the steps
        """
        grid_output = grid_output.permute(1, 0)
        grid_output = grid_output.view(self.output_channels_count, self.sample_height, self.sample_width)
        return grid_output

    def to(self, *args, **kwargs):
        self = super(GraphCastNetBaseline, self).to(*args, **kwargs)

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.g2m_graph = self.g2m_graph.to(device)
        self.m2m_graph = self.m2m_graph.to(device)
        self.m2g_graph = self.m2g_graph.to(device)

        self.m2m_node_input = self.m2m_node_input.to(*args, **kwargs)
        self.g2m_edge_input = self.g2m_edge_input.to(*args, **kwargs)
        self.m2m_edge_input = self.m2m_edge_input.to(*args, **kwargs)
        self.m2g_edge_input = self.m2g_edge_input.to(*args, **kwargs)

        return self

    @property
    def generated_channels_count(self):
        # !!! The following quantities need to be kept aligned with the dataloader which produces this data !!!
        return \
            int(self._include_static_data) * 2 + \
            int(self._include_spatial_info) * 3 + \
            int(self._include_temporal_info) * 4 + \
            int(self._include_solar_radiation)

    @property
    def input_channels_count(self):
        return self.sample_channels + self.generated_channels_count

    @property
    def output_channels_count(self):
        # !!! The following quantity need to be kept aligned with the dataloader !!!
        return self.sample_channels

    @property
    def includes_static_data(self):
        return self._include_static_data

    @property
    def includes_spatial_info(self):
        return self._include_spatial_info

    @property
    def includes_temporal_info(self):
        return self._include_temporal_info

    @property
    def includes_solar_radiation(self):
        return self._include_solar_radiation
    