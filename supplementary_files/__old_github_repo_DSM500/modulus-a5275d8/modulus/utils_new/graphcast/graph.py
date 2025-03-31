# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 schups@gmail.com (Modifications)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import os

from modulus.utils.graphcast.icosahedral_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
    faces_to_edges,
    merge_meshes
)

from modulus.utils.graphcast.graph_utils import (
    create_graph,
    create_heterograph,
    max_edge_length,
    latlon2xyz,
    xyz2latlon,
    geospatial_rotation,
    azimuthal_angle,
    polar_angle
)

from modulus.utils_new.caching import Cache, MeshesCacheGuard

class Graph:
    def __init__(self, list_grid_3D_coords, list_grid_latlon, mesh_level=6, cache_enabled=True):
        self.cache_enabled = cache_enabled
        
        self.list_grid_3D_coords = list_grid_3D_coords # e.g. (1038240, 3)
        self.list_grid_latlon = list_grid_latlon # e.g. (1038240, 2)

        # build the multi-mesh
        _list_of_meshes = self._get_hierarchy_of_meshes(splits=mesh_level)
        _finest_mesh = _list_of_meshes[-1]
        _multimesh = merge_meshes(_list_of_meshes)

        _finest_mesh_src, _finest_mesh_dst = faces_to_edges(_finest_mesh.faces)
        _multimesh_src, _multimesh_dst = faces_to_edges(_multimesh.faces)

        self.finest_mesh_src = torch.tensor(_finest_mesh_src, dtype=torch.int32)
        self.finest_mesh_dst = torch.tensor(_finest_mesh_dst, dtype=torch.int32)
        self.multimesh_src = torch.tensor(_multimesh_src, dtype=torch.int32)
        self.multimesh_dst = torch.tensor(_multimesh_dst, dtype=torch.int32)
        self.faces = torch.tensor(_multimesh.faces, dtype=torch.int32)
        self.vertices = torch.tensor(_multimesh.vertices, dtype=torch.float32) # i.e. 3D coords, shape for 6 splits: (40962, 3)
        self.vertices_latlon = xyz2latlon(self.vertices) # i.e. lat/lon coords, shape for 6 splits: (40962, 2)

    def log(self, *args, **kwargs):
        from modulus.distributed import DistributedManager
        dist = DistributedManager()
        if dist.rank == 0:
            print(dist.rank, *args)

    def sample(self, a):
        _a = a.to(dtype=torch.float32)
        _a = _a.flatten()[:3].cpu().detach().numpy()
        return str(f"{list(_a)} @ {str(a.device)}")

    def create_m2m_graph(self):
        """Create the multimesh graph."""
        m2m_graph = create_graph(
            self.multimesh_src,
            self.multimesh_dst,
            to_bidirected=True,
            add_self_loop=False
        )
        m2m_graph.ndata["spatial_info"] = self._compute_vertices_spatial_info(self.vertices_latlon)
        m2m_graph.edata["displacement_info"] = self._compute_edge_displacement(
            graph=m2m_graph, 
            src_3D_coords=self.vertices, 
            dst_3D_coords=self.vertices
        )
        return m2m_graph

    def create_g2m_graph(self):
        """Create the grid2mesh bipartite graph."""

        # Get max edge length for the finest mesh, e.g. 0.020673068 for 7 mesh levels (splits=6)
        max_edge_len = max_edge_length(self.vertices, self.finest_mesh_src, self.finest_mesh_dst)

        src, dst = self._get_g2m_edges(self.list_grid_3D_coords, self.vertices, max_edge_len)

        g2m_graph = create_heterograph(src, dst, ("grid", "g2m", "mesh"))
        g2m_graph.edata["displacement_info"] = self._compute_edge_displacement(
            graph=g2m_graph, 
            src_3D_coords=self.list_grid_3D_coords, 
            dst_3D_coords=self.vertices
        )
        return g2m_graph

    def create_m2g_graph(self):
        """Create the mesh2grid bipartite graph."""

        src, dst = self._get_m2g_edges(self.vertices, self.faces, self.list_grid_3D_coords)
        
        m2g_graph = create_heterograph(src, dst, ("mesh", "m2g", "grid"))
        m2g_graph.edata["displacement_info"] = self._compute_edge_displacement(
            graph=m2g_graph, 
            src_3D_coords=self.vertices, 
            dst_3D_coords=self.list_grid_3D_coords
        )
        return m2g_graph

    def _get_g2m_edges(self, grid_coords: torch.Tensor, mesh_coords: torch.Tensor, max_edge_len: float):
        # For each grid 3D coordinate, find the nearest mesh verties
        n_neighbors = 4
        neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(mesh_coords) # mesh 3D coordinates
        distances, indices = neighbors.kneighbors(grid_coords) # grid 3D coordinates

        distances = torch.tensor(distances, dtype=torch.float32)
        mask = (distances <= 0.6 * max_edge_len).reshape(-1)
        src = torch.arange(grid_coords.shape[0], dtype=torch.int32).unsqueeze(1).expand(-1, n_neighbors).reshape(-1)
        dst = torch.tensor(indices, dtype=torch.int32).reshape(-1)

        src = src[mask]
        dst = dst[mask]

        return src, dst
    
    def _get_m2g_edges(self, vertices: torch.Tensor, faces: torch.Tensor, grid_coords: torch.Tensor):
        face_centroids = vertices[faces].mean(dim=1)

        # For each grid 3D coordinate, find the nearest mesh face
        neighbors = NearestNeighbors(n_neighbors=1).fit(face_centroids) # face centroid vectors 
        _, indices = neighbors.kneighbors(grid_coords) # grid coordinates
        indices = indices.squeeze()

        src = faces[indices].view(-1)
        dst = torch.arange(len(indices), dtype=torch.int32).repeat_interleave(3)
        return src, dst

    def _get_hierarchy_of_meshes(self, splits):
        if self.cache_enabled:
            cache = Cache()
            if cache.is_cached(Cache.MESHES):
                return cache.load(Cache.MESHES, guards=[MeshesCacheGuard(splits+1)])
            else:
                meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
                cache.store(Cache.MESHES, meshes)
                return meshes
        else:
            meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
            return meshes

    def _compute_vertices_spatial_info(self, vertices_latlon):
        spatial_info = torch.stack((
            #torch.sin(vertices_latlon[:, 0]), # TODO: Reenable this
            torch.cos(vertices_latlon[:, 0]),
            torch.sin(vertices_latlon[:, 1]),
            torch.cos(vertices_latlon[:, 1])
        ), dim=-1)

        return spatial_info

    def _compute_edge_displacement(self, graph, src_3D_coords, dst_3D_coords):
        """
        Computes the normalized displacement vector between source and destination nodes.
        This is done in a local coordinate system where the destination node is aligned with the x-axis.
        # TODO: Verify if this function behaves as expected... I can't visually interprest the direction of the displacement nodes
        """
        src, dst = graph.edges()
        src_3D_coord = src_3D_coords[src]
        dst_3D_coord = dst_3D_coords[dst]

        # azimuthal & polar rotation
        dst_latlon_rad = xyz2latlon(dst_3D_coord, unit="rad")
        dst_lat_rad, dst_lon_rad = dst_latlon_rad[:, 0], dst_latlon_rad[:, 1]
        theta_azimuthal = azimuthal_angle(dst_lon_rad)
        theta_polar = polar_angle(dst_lat_rad)

        # First part of the rotation
        src_3D_coord = geospatial_rotation(src_3D_coord, theta=theta_azimuthal, axis="z", unit="rad")
        dst_3D_coord = geospatial_rotation(dst_3D_coord, theta=theta_azimuthal, axis="z", unit="rad")
        assert dst_3D_coord[:, 1].max() < 1e-5 # Adopt original code tolerance

        # Second part of the rotation
        src_3D_coord = geospatial_rotation(src_3D_coord, theta=theta_polar, axis="y", unit="rad")
        dst_3D_coord = geospatial_rotation(dst_3D_coord, theta=theta_polar, axis="y", unit="rad")
        assert torch.abs(dst_3D_coord[:, 0] - 1.0).max() < 1e-5
        assert dst_3D_coord[:, [1,2]].max() < 1e-5

        # Compute displacement
        displacement = src_3D_coord - dst_3D_coord
        displacement_norm = torch.norm(displacement, dim=-1, keepdim=True)
        # Normalize
        max_displacement_norm = displacement_norm.max()
        displacement /= max_displacement_norm
        displacement_norm /= max_displacement_norm

        displacement_info = torch.cat((displacement, displacement_norm), dim=-1)
        return displacement_info
