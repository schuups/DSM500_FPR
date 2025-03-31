# Snippets

## Add venv to jupyter
`python -m ipykernel install --user --name=venv_x86`

## Start jupyter
jupyter lab --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.ip='0.0.0.0' --NotebookApp.port=8888 --NotebookApp.open_browser=False --notebook-dir=$(pwd)

## Run training from console
```sh
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500 # default from torch launcher

source /iopsstor/scratch/cscs/stefschu/DSM500/gitlab/venv_arm64/bin/activate
cd /iopsstor/scratch/cscs/stefschu/DSM500/github/examples/weather/graphcast
python train_graphcast.py
```