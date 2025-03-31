import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import nvidia.dali as dali
import nvidia.dali.plugin.pytorch as dali_pth

import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

from modulus.distributed import DistributedManager

SEED = 0 # TODO


def log(caller, msg):
    dist = DistributedManager()
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Rank {dist.rank}][{time}][{caller}] {msg}", flush=True)

class Datapipe:
    def __init__(self, dist):
        log(f"{self.__class__.__name__}.__init__", "Datapipe creation")
        self.dist = dist
        self.pipe = None
        self.ds = np.array(range(1000, 1017), dtype=np.float32)

    def _create_pipeline(self) -> dali.Pipeline:
        log(f"{self.__class__.__name__}._create_pipeline", "Pipeline creation")

        pipe = dali.Pipeline(
            batch_size=1,
            num_threads=4,
            prefetch_queue_depth=2,
            py_num_workers=3,
            device_id=self.dist.device.index,
            py_start_method="spawn",
            set_affinity=True
        )
        
        with pipe:
            source = ExternalSource(dist=self.dist, ds=self.ds)
            data, sample_info = dali.fn.external_source(
                source,
                num_outputs=2,
                parallel=True,
                batch=False
            )
            pipe.set_outputs(data, sample_info)

        return pipe

    def __iter__(self):
        log(f"{self.__class__.__name__}.__iter__", "Iterator creation")
        if self.pipe is None:
            self.pipe = self._create_pipeline()
        self.pipe.reset()
        generator = dali_pth.DALIGenericIterator([self.pipe], ["data", "sample_info"])
        return generator

    def __len__(self):
        #log(f"{self.__class__.__name__}.__len__", f"Returning {len(self.ds)}")
        return len(self.ds)

class ExternalSource:
    def __init__(self, dist, ds):
        self.dist = dist
        self.ds = ds
        self.last_epoch = None
    
    def __call__(self, sample: dali.types.SampleInfo):
        if self.last_epoch != sample.epoch_idx:
            self.indices = np.arange(len(self))
            np.random.default_rng(seed=SEED + sample.epoch_idx).shuffle(self.indices)
            self.indices = np.array_split(self.indices, self.dist.world_size)
            self.indices = self.indices[self.dist.rank]
            log(f"{self.__class__.__name__}.__init__", f"\tReshuffling! Indices: {self.indices}")

            self.last_epoch = sample.epoch_idx

        if sample.idx_in_epoch >= len(self.indices):
            log(f"{self.__class__.__name__}.__call__", f"\tStopIteration: {sample.idx_in_epoch=}, {len(self.indices)=}")
            raise StopIteration()
        
        idx = self.indices[sample.idx_in_epoch]
        data = self.ds[idx]
        log(f"{self.__class__.__name__}.__call__", f"\tInvoked ({sample.idx_in_epoch=} in {self.indices=} gives {idx=} ---> {data})")

        output = (
            np.array([data]),
            np.array([sample.idx_in_epoch, sample.epoch_idx, idx], dtype=np.float32)
        )
        return output

    # def __len__(self):
    #     log(__class__.__name__, f"__len__ called, returning {len(self.ds)}")
    #     return len(self.ds)

if __name__ == "__main__":
    DistributedManager.initialize()
    dist = DistributedManager()

    dp = Datapipe(dist)
    log(__name__, f"Datapipe created, has length {len(dp)}")
    torch.distributed.barrier()
    for validation_i in range(3):
        log(__name__, "-"*80)
        log(__name__, f"Starting {validation_i=}")
        torch.distributed.barrier()
        for iteration_i, sample in enumerate(dp):
            data = int(sample[0]["data"].squeeze(0).tolist()[0])
            sinfo = [int(v) for v in sample[0]["sample_info"].squeeze(0).tolist()]
            log(__name__, f"--> {validation_i=} {iteration_i=} got {data} with [sample.idx_in_epoch={sinfo[0]}, sample.epoch_idx={sinfo[1]}, idx={sinfo[2]}]")
        log(__name__, f"Completed {validation_i=}")
        torch.distributed.barrier()

