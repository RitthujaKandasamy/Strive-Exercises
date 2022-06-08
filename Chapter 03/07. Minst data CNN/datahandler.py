from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)





# Define a transform to normalize the data (Preprocessing) and cast to tensor
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True,
                          train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True,
                         train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


print(trainloader.dataset)
print(testloader.dataset)