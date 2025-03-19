from mpi4py import MPI

import os
import h5py
import numpy as np
import glob
import torch
import sys
import xarray as xr
from netCDF4 import Dataset as ncDataset
from collections import defaultdict

def get_sorted_file_paths(base_path='/iopsstor/scratch/cscs/stefschu/DSM500_FPR/data/FCN_ERA5_data_v0'):
    h5_files = glob.glob(os.path.join(base_path, '**', '*.h5'), recursive=True)
    h5_files = [f for f in h5_files if "out_of_sample" not in f]
    return sorted(h5_files)

def _slice_hdf5(path, channel):
    with h5py.File(path, 'r') as h5_file:
        _data = np.array(h5_file["fields"][:, channel, :, :], dtype=np.float64)

        if channel == 20:
            _filler = np.array(h5_file["fields"][:, 2, :, :], dtype=np.float64)
            _mask = _data == -32767.0
            _data[_mask] = _filler[_mask]
        return _data

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Rank of the current process
    size = comm.Get_size()  # Total number of processes
    assert size == 39 * 21 # 39 files, 21 channels

    # Translate parameters
    file_path_id = rank // 21
    file_path = get_sorted_file_paths()[file_path_id]
    channel_id = rank % 21
    print(f"Rank {rank:03} processing channel {channel_id:02} of {file_path_id=} {file_path}")

    # Compute mean
    local_channel_data = _slice_hdf5(file_path, channel_id)
    assert local_channel_data.shape[0] == 1460 # To assure I can take the mean of means
    if channel_id == 20:
        local_channel_mean = np.nanmean(local_channel_data, dtype=np.float128)
    else:
        local_channel_mean = np.mean(local_channel_data, dtype=np.float128)

    # Gather results at rank 0
    results = comm.gather({
        "channel_id": channel_id,
        "local_channel_mean": local_channel_mean,
        "file_path": file_path
    }, root=0)

    if rank == 0:
        all_means = defaultdict(list)

        print(f"Individual results:")
        for result in results:
            print(result)
            all_means[result["channel_id"]].append(result["local_channel_mean"])
        print()

        print("Aggregated means:")
        aggregated_means = np.array(list(all_means.values()))
        print(aggregated_means.shape, aggregated_means.dtype)
        print(aggregated_means)
        print()

        print("Global channel means:")
        global_means = np.mean(aggregated_means, axis=1)
        print(global_means.shape, global_means.dtype)
        print(global_means)

        global_means = global_means.astype(np.float32)
        global_means = global_means.reshape(1, -1, 1, 1)
        #np.save(f"/iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/00_stat_files_creation/global_means.npy", global_means)
        np.save(f"/iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/improvements/00_sst_fix/data/global_means_with_sst_fix.npy", global_means)
        print()

        del aggregated_means 
        del global_means

    comm.barrier()    

    # Compute stddev
    #global_means = np.load(f"/iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/00_stat_files_creation/stats_files_recreation/global_means_without_sst_fix.npy")
    global_means = np.load(f"/iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/improvements/00_sst_fix/data/global_means_with_sst_fix.npy")
    channel_mean = global_means.squeeze()[channel_id]

    local_parts = list()
    if channel_id == 20:
        local_parts.append(np.nansum(np.square(local_channel_data[:300] - channel_mean), dtype=np.float128))
        local_parts.append(np.nansum(np.square(local_channel_data[300:600] - channel_mean), dtype=np.float128))
        local_parts.append(np.nansum(np.square(local_channel_data[600:900] - channel_mean), dtype=np.float128))
        local_parts.append(np.nansum(np.square(local_channel_data[900:1200] - channel_mean), dtype=np.float128))
        local_parts.append(np.nansum(np.square(local_channel_data[1200:] - channel_mean), dtype=np.float128))
    else:
        local_parts.append(np.sum(np.square(local_channel_data[:300] - channel_mean), dtype=np.float128))
        local_parts.append(np.sum(np.square(local_channel_data[300:600] - channel_mean), dtype=np.float128))
        local_parts.append(np.sum(np.square(local_channel_data[600:900] - channel_mean), dtype=np.float128))
        local_parts.append(np.sum(np.square(local_channel_data[900:1200] - channel_mean), dtype=np.float128))
        local_parts.append(np.sum(np.square(local_channel_data[1200:] - channel_mean), dtype=np.float128))
    local_parts = np.array(local_parts)
    local_sum = np.sum(local_parts)
    
    if channel_id == 20:
        summed_elements = np.sum(~np.isnan(local_channel_data), dtype=np.float128)
    else:
        summed_elements = np.float128(local_channel_data.size)

    # Gather results at rank 0
    results = comm.gather({
        "channel_id": channel_id,
        "local_sum": local_sum,
        "summed_elements": summed_elements,
        "file_path": file_path
    }, root=0)

    if rank == 0:
        local_sums = defaultdict(list)
        local_elements = defaultdict(list)
        
        print(f"Individual results:")
        for result in results:
            print(result)
            local_sums[result["channel_id"]].append(result["local_sum"])
            local_elements[result["channel_id"]].append(result["summed_elements"])

        print("Aggregated sums:")
        aggregated_sums = np.array(list(local_sums.values()))
        aggregated_elements = np.array(list(local_elements.values()))

        print(aggregated_sums.shape, aggregated_sums.dtype)
        print(aggregated_sums)
        print(aggregated_elements.shape, aggregated_elements.dtype)
        print(aggregated_elements)
        print()

        print("Global channel standard deviations:")
        global_stddevs = np.sqrt(np.sum(aggregated_sums, axis=1) / np.sum(aggregated_elements, axis=1))
        print(global_stddevs.shape, global_stddevs.dtype)
        print(global_stddevs)

        global_stddevs = global_stddevs.astype(np.float32)
        global_stddevs = global_stddevs.reshape(1, -1, 1, 1)
        #np.save(f"/iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/00_stat_files_creation/global_stds_without_sst_fix.npy", global_stddevs)
        np.save(f"/iopsstor/scratch/cscs/stefschu/DSM500_FPR/supplementary_files/improvements/00_sst_fix/data/global_stds_with_sst_fix.npy", global_stddevs)

if __name__ == "__main__":
    main()