import argparse
import os
import sys
import tempfile
from time import sleep
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    while True:
        print(f"[pid: {os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
        sleep(1)


def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[pid: {os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=60))
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
