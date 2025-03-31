import torch
from dgl import DGLGraph
from torch import Tensor
from typing import Tuple
import dgl.function as fn

def aggregate_and_concat(
    edge_feats: Tensor,
    node_feats: Tensor,
    aggregation_op: str,
    graph: DGLGraph,
):
    """
    Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.
    """
    with graph.local_scope():
        # populate features on graph edges
        graph.edata["x"] = edge_feats

        # aggregate edge features
        if aggregation_op == "sum":
            graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
        elif aggregation_op == "mean":
            graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
        else:
            raise RuntimeError("Not a valid aggregation!")

        h_dest = graph.dstdata.pop("h_dest")
        
        return torch.cat((h_dest, node_feats), dim=-1)

@torch.jit.script
def sum_efeat_dgl(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
) -> Tensor:
    """
    Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    src_feat : Tensor
        Source node features.
    dst_feat : Tensor
        Destination node features.
    src_idx : Tensor
        Source node indices.
    dst_idx : Tensor
        Destination node indices.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """
    efeat.add_(src_feat[src_idx]).add_(dst_feat[dst_idx])
    return efeat
    #return efeat + src_feat[src_idx] + dst_feat[dst_idx]

def sum_efeat(
    edge_feats: Tensor,
    node_feats: Tuple[Tensor],
    graph: DGLGraph,
):
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features (static setting) or tuple of node features of
        source and destination nodes (bipartite setting).
    graph : DGLGraph | CuGraphCSC
        The underlying graph.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """

    src, dst = graph.edges()
    sum_efeat = sum_efeat_dgl(edge_feats, node_feats[0], node_feats[1], src, dst)

    return sum_efeat