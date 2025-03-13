# Setup notes

## Download data from CDS
```sh
cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/baseline/03_static_files_download/data

# Query CDS as per printscreens

wget -O land_sea_mask.nc https://object-store.os-api.cci2.ecmwf.int/cci2-prod-cache/e0265a1bf89a9098d29c87eda964e086.nc
mv e0265a1bf89a9098d29c87eda964e086.nc land_sea_mask.nc

wget -O geopotential.nc https://object-store.os-api.cci2.ecmwf.int/cci2-prod-cache/8a2fe8bcf6f938cf4c33c43dd45a9505.nc
mv 8a2fe8bcf6f938cf4c33c43dd45a9505.nc geopotential.nc
```

## Install data
```sh
mkdir -p /iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0/static
cd /iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0/static

ln -s /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/baseline/03_static_files_download/data/geopotential.nc geopotential.nc
ln -s /iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/baseline/03_static_files_download/data/land_sea_mask.nc land_sea_mask.nc
```