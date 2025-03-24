import sys

sys.path.insert(0, "/iopsstor/scratch/cscs/stefschu/DSM500_FPR/modulus-baseline")

from modulus.distributed.manager import DistributedManager

if __name__ == "__main__":
    DistributedManager.initialize()
    print(DistributedManager().rank)
    print(DistributedManager().world_size)

    DistributedManager.cleanup()