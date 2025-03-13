# Setup notes

## Copying what necessary from the modulus codebase
Files are carefully copied from the original modulus cloned files. This is done to select just what is required, thus supporting a consise base for interested readers to explore.

```sh
ORIGINAL_BASE_PATH=/iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-a5275d8

cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline

cp $ORIGINAL_BASE_PATH/examples/weather/graphcast/train_graphcast.py .

# iteratively run python train_graphcast.py until there are not more dependencies missing

mkdir -p modulus/models/graphcast
mkdir -p modulus/models/gnn_layers
mkdir -p modulus/models/layers
mkdir -p modulus/models/registry
mkdir -p modulus/utils/graphcast
mkdir -p modulus/launch/logging
mkdir -p modulus/launch/utils
mkdir -p modulus/datapipes/climate
mkdir -p modulus/distributed
mkdir -p modulus/utils/graphcast
mkdir -p modulus/datapipes/climate/utils
mkdir loss
mkdir conf

cp $ORIGINAL_BASE_PATH/modulus/models/graphcast/graph_cast_net.py modulus/models/graphcast/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/embedder.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/mesh_graph_mlp.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/utils.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/__init__.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/distributed_graph.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/distributed/__init__.py modulus/distributed/
cp $ORIGINAL_BASE_PATH/modulus/distributed/autograd.py modulus/distributed/
cp $ORIGINAL_BASE_PATH/modulus/distributed/utils.py modulus/distributed/
cp $ORIGINAL_BASE_PATH/modulus/distributed/manager.py modulus/distributed/
cp $ORIGINAL_BASE_PATH/modulus/distributed/config.py modulus/distributed/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/graph.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/mesh_graph_decoder.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/mesh_graph_encoder.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/layers/__init__.py modulus/models/layers/
# comment out in everything other than activations.py
cp $ORIGINAL_BASE_PATH/modulus/models/layers/activations.py modulus/models/layers/
cp $ORIGINAL_BASE_PATH/modulus/models/meta.py modulus/models/
cp $ORIGINAL_BASE_PATH/modulus/models/module.py modulus/models/
cp $ORIGINAL_BASE_PATH/modulus/registry/__init__.py modulus/registry/
cp $ORIGINAL_BASE_PATH/modulus/registry/model_registry.py modulus/registry/
cp $ORIGINAL_BASE_PATH/modulus/utils/filesystem.py modulus/utils/
cp $ORIGINAL_BASE_PATH/modulus/utils/graphcast/graph.py modulus/utils/graphcast/
cp $ORIGINAL_BASE_PATH/modulus/utils/graphcast/graph_utils.py modulus/utils/graphcast/
cp $ORIGINAL_BASE_PATH/modulus/utils/graphcast/icosahedral_mesh.py modulus/utils/graphcast/
cp $ORIGINAL_BASE_PATH/modulus/models/graphcast/graph_cast_processor.py modulus/models/graphcast/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/mesh_edge_block.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/models/gnn_layers/mesh_node_block.py modulus/models/gnn_layers/
cp $ORIGINAL_BASE_PATH/modulus/utils/graphcast/loss.py modulus/utils/graphcast/
cp $ORIGINAL_BASE_PATH/modulus/launch/logging/console.py modulus/launch/logging/
cp $ORIGINAL_BASE_PATH/modulus/launch/logging/__init__.py modulus/launch/logging/
cp $ORIGINAL_BASE_PATH/modulus/launch/logging/launch.py modulus/launch/logging/
cp $ORIGINAL_BASE_PATH/modulus/launch/logging/wandb.py modulus/launch/logging/
cp $ORIGINAL_BASE_PATH/modulus/launch/logging/utils.py modulus/launch/logging/
cp $ORIGINAL_BASE_PATH/modulus/launch/logging/mlflow.py modulus/launch/logging/
cp $ORIGINAL_BASE_PATH/modulus/launch/utils/__init__.py modulus/launch/utils/
cp $ORIGINAL_BASE_PATH/modulus/launch/utils/checkpoint.py modulus/launch/utils/
cp $ORIGINAL_BASE_PATH/modulus/utils/capture.py modulus/utils/
cp $ORIGINAL_BASE_PATH/examples/weather/graphcast/train_utils.py .
cp $ORIGINAL_BASE_PATH/examples/weather/graphcast/loss/utils.py loss/
cp $ORIGINAL_BASE_PATH/examples/weather/graphcast/train_base.py .
cp $ORIGINAL_BASE_PATH/examples/weather/graphcast/validation_base.py .
cp $ORIGINAL_BASE_PATH/modulus/datapipes/climate/__init__.py modulus/datapipes/climate/
# comment out climate and synthetic from the __init__.py file
cp $ORIGINAL_BASE_PATH/modulus/datapipes/climate/era5_hdf5.py modulus/datapipes/climate/
cp $ORIGINAL_BASE_PATH/modulus/datapipes/climate/utils/invariant.py modulus/datapipes/climate/utils/
cp $ORIGINAL_BASE_PATH/modulus/datapipes/climate/utils/zenith_angle.py modulus/datapipes/climate/utils/
cp $ORIGINAL_BASE_PATH/modulus/datapipes/datapipe.py modulus/datapipes/
cp $ORIGINAL_BASE_PATH/modulus/datapipes/meta.py modulus/datapipes/
cp $ORIGINAL_BASE_PATH/modulus/datapipes/climate/synthetic.py modulus/datapipes/climate/
cp $ORIGINAL_BASE_PATH/modulus/utils/graphcast/data_utils.py modulus/utils/graphcast/
cp $ORIGINAL_BASE_PATH/examples/weather/graphcast/conf/config.yaml conf/
cp $ORIGINAL_BASE_PATH/modulus/models/layers/fused_silu.py modulus/models/layers/
cp $ORIGINAL_BASE_PATH/modulus/models/__init__.py modulus/models/
cp $ORIGINAL_BASE_PATH/modulus/__init__.py modulus/
```

## Running the code

### At CLI

It could be that for development purposes is necessary to run a training from cli (single node).

```sh
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train_graphcast.py
```

### Via workload manager

```sh
sbatch train_graphcast.sbatch
```