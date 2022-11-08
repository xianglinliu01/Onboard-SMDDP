import torch
import torch.distributed as dist
import os
import sys
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class DistributedDataParallel(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self.hook_fun)
            #dist.broadcast(param, 0)

    def forward(self, x):
        return self.module.forward(x)

    def hook_fun(self, grad):
        return 0.1 * grad

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)


if __name__ == '__main__':

    dist.init_process_group(backend="nccl")

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(rank)
    model = BasicNet().to(device)
    model = DistributedDataParallel(model) 
    print("device id is ", rank)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    num_epochs = 1
    for epoch in range(1, num_epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        #if rank == 0:
        print("the loss is: ", loss)

    dist.destroy_process_group()