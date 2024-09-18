import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
torch.use_deterministic_algorithms(True) # ensure reproducibility of experiments


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__() # call the parent class constructor (nn.Module)
        # 3 input image channel, 6 output channels, 5x5 kernels
        self.cn1 = nn.Conv2d(3,6,5)
        # 6 input image channel, 16 output channels, 5x5 kernels
        self.cn2 = nn.Conv2d(6,16,5)
        # fully connected layers of size 120, 84, 10
        # 5 x 5 is the spatial dimension of at this layer (going into fc1) (kernel size)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        # Convolution with 5x5 kernel
        x = F.relu(self.cn1(x))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))
        # Convolution with 5x5 kernel
        x = F.relu(self.cn2(x))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2,2))
        # Flatten the image into a single vector
        x = x.view(-1, self.flattened_features(x))
        # Fully connected operations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def flattened_features(self, x):
        """
        Determining the number of features present in the tensor so that it can be properly flattened
        """
        # all except the first batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
def train(net, trainloader, optim, epoch):
    # initialize the loss
    loss_total = 0.0
    for i, data in enumerate(trainloader, 0):
        ip, ground_truth = data
        optim.zero_grad()
        op = net(ip)
        loss = nn.CrossEntropyLoss()(op, ground_truth)
        loss.backward()
        optim.step()
        loss_total += loss.item()
        # Print loss every 1000 mini-batches
        if (i+1) % 1000 == 0:
            print(f'Epoch: {epoch + 1}, Mini-batch: {i+1:5d}, Loss: {loss_total/100:.3f}%')
            loss_total = 0.0 # reset loss for mini-batch (1000)
        
lenet = LeNet()
print(lenet)
        