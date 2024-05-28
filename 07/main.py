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

def inference(model, device_id, test_loader, rank):
    print("inferencing......")
    model.eval()
    size = torch.tensor(0.).to(device_id)
    correct = torch.tensor(0.).to(device_id)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device_id)
            labels = labels.to(device_id)
            outputs = model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()
            if (i + 1) % 100 == 0:
                print(f"images: {images}, labels: {labels}, scores: {outputs}")

    dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    # if rank == 0:
    #     print('Evaluate accuracy is {:.2f}'.format(correct / size))
    print('Evaluate accuracy is {:.2f}'.format(correct / size))
    print("evaluate done")


def evaluate(model, device_id, test_loader, rank):
    print("evaluating......")
    model.eval()
    size = torch.tensor(0.).to(device_id)
    correct = torch.tensor(0.).to(device_id)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device_id)
            labels = labels.to(device_id)
            outputs = model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()

    dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    # if rank == 0:
    #     print('Evaluate accuracy is {:.2f}'.format(correct / size))
    print('Evaluate accuracy is {:.2f}'.format(correct / size))
    print("evaluate done")


def train(ddp_model, loss_fn, optimizer, device_id, train_loader, rank):
    ckpt_path = 'tmp.pth'
    epochs = 2

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader) if rank == 0 else train_loader):
            images = images.to(device_id)
            labels = labels.to(device_id)
            outputs = ddp_model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            if (i + 1) % 100 == 0 and rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.
                      format(epoch + 1, epochs, i + 1, total_step, loss.item()))
        evaluate(ddp_model, device_id, train_loader, rank)

    if rank == 0:
        print("Training complete in: " + str(datetime.now() - start))
        torch.save(ddp_model.state_dict(), ckpt_path)
        print("ckpt done.")


def main():
    batch_size = 4
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    pid = os.getpid()
    device_count = torch.cuda.device_count()
    print(f'current pid: {pid}; current rank {rank}; total device count: {device_count}')
    device_id = rank % device_count

    model = Model().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.CrossEntropyLoss().to(device_id)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    train(ddp_model, loss_fn, optimizer, device_id, train_loader, rank)


def main01():
    ckpt_path = 'tmp.pth'
    epochs = 2
    batch_size = 4
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    pid = os.getpid()
    device_count = torch.cuda.device_count()
    print(f'current pid: {pid}; current rank {rank}; total device count: {device_count}')
    device_id = rank % device_count

    model = Model().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.CrossEntropyLoss().to(device_id)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader) if rank == 0 else train_loader):
            images = images.to(device_id)
            labels = labels.to(device_id)
            outputs = ddp_model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            if (i + 1) % 100 == 0 and rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.
                      format(epoch + 1, epochs, i + 1, total_step, loss.item()))
        evaluate(model, device_id, train_loader, rank)

    if rank == 0:
        print("Training complete in: " + str(datetime.now() - start))
        torch.save(ddp_model.state_dict(), ckpt_path)
        print("ckpt done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
