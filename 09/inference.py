import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learn rate")
    parser.add_argument('--epochs', type=int, default=5, help="train epoch")
    parser.add_argument('--ckpt_path', type=str, default="model.pt", help="ckpt path")
    parser.add_argument('--pre_ckpt_path', type=str, default="/workspace/workflow/model.pt", help="pretrained ckpt path")
    parser.add_argument('--train_data_dir', type=str, default="/workspace/workflow/data/", help="train data dir")   
    
    args = parser.parse_args()

    return args

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def inference():
    print("inferencing......")
    args = parse_args()
    batch_size = args.bs
    train_data_dir = args.train_data_dir
    pre_ckpt_path = args.pre_ckpt_path

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    pid = os.getpid()
    device_count = torch.cuda.device_count()
    print(f'current pid: {pid}; current rank {rank}; total device count: {device_count}')
    device_id = rank % device_count

    model = Model()
    model.load_state_dict(torch.load(pre_ckpt_path))
    model = Model().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.CrossEntropyLoss().to(device_id)

    test_dataset = torchvision.datasets.MNIST(root=train_data_dir,
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=test_sampler)

    
    ddp_model.eval()
    size = torch.tensor(0.).to(device_id)
    correct = torch.tensor(0.).to(device_id)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device_id)
            labels = labels.to(device_id)
            outputs = ddp_model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()
            if (i + 1) % 100 == 0:
                print(f"images: {images}, labels: {labels}, scores: {outputs}")

    dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    
    print('Evaluate accuracy is {:.2f}'.format(correct / size))
    print("evaluate done")


if __name__ == "__main__":
    inference()
