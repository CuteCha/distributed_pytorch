import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def main():
    ckpt_path = 'tmp.pth'
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    pid = os.getpid()
    device_count = torch.cuda.device_count()
    print(f'current pid: {pid}; current rank {rank}; total device count: {device_count}')
    device_id = rank % device_count
    model = Model().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    if rank == 0:
        torch.save(ddp_model.state_dict(), ckpt_path)
    
    dist.barrier()
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()