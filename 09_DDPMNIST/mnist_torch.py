import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

LOG_DIR = os.path.join(os.getcwd(), "logs")

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#device = "cuda"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

import os
import torch.distributed as dist

# def setup(rank, world_size):
#     "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
#     os.environ["MASTER_ADDR"] = 'localhost'
#     os.environ["MASTER_PORT"] = "12355"

#     # Initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup():
#     "Cleans up the distributed environment"
#     dist.destroy_process_group()

from torch.nn.parallel import DistributedDataParallel as DDP

def train():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(rank)
    model = BasicNet().to(device)
    model = DDP(model, device_ids=[rank]) 
    print("device id is ", rank)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Train for a few epochs
    model.train()
    num_epochs = 2
    for epoch in range(1, num_epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print("the loss is: ", loss)

def train_naive_ddp():
    # rank = int(os.environ["LOCAL_RANK"])
    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    device = torch.device(rank)
    model = BasicNet().to(device)
    print("device id is ", rank)
    model.train()
    num_epochs = 2
    optimizer = optim.SGD(model.parameters(),\
                          lr=0.01, momentum=0.5)
    for epoch in range(1, num_epochs+1):
        print(f"running epoch {epoch} of {num_epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            # set the grad to zero (in place of optimizer.zero_grad())
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                param.grad.data /= world_size
            optimizer.step()
    
if __name__ == "__main__":
    os.system("rm -rf %s" %(LOG_DIR))
    os.system("mkdir -p %s" %(LOG_DIR))
    torch.distributed.init_process_group(backend="nccl")
    #train()
    train_naive_ddp()
    dist.destroy_process_group()

