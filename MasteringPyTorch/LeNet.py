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
        op = net(ip) # output probabilities
        loss = nn.CrossEntropyLoss()(op, ground_truth)
        loss.backward()
        optim.step()
        loss_total += loss.item()
        # Print loss every 1000 mini-batches
        if (i+1) % 1000 == 0:
            print(f'[Epoch: {epoch + 1}, Mini-batch: {i+1:5d}] Loss: {loss_total/1000:.3f}')
            loss_total = 0.0 # reset loss for mini-batch (1000)

def test(net, testloader):
    success  = 0
    counter = 0
    with torch.no_grad():
        for data in testloader:
            im, ground_truth = data
            op = net(im) # output probabilities
            _, pred = torch.max(op.data, 1)
            counter += ground_truth.size(0) # because processing batches
            success += (pred == ground_truth).sum().item()
    print(f"LeNet accuracy on 10,000 images from test dataset {100 * success/counter:.2f}%") 
    
def imageshow(image):
    npimage = image/2 + 0.5 # unnormalize
    plt.imshow(np.transpose(npimage, (1,2,0)))
    plt.show()
    

if __name__ == '__main__':       
    lenet = LeNet()
    # print(lenet)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32,4),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    num_images = 4
    imageshow(torchvision.utils.make_grid(images[:num_images]))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(num_images)))
    
    # Training LeNet
    optim = torch.optim.Adam(lenet.parameters(), lr=0.001)
    # Loop over the dataset multiple times
    for epoch in range(50):
        train(lenet, trainloader, optim, epoch)
        print()
        test(lenet, testloader)
        print()
    print('Finished Training')
    
    # Save the model
    model_path = './cifar_model.pth'
    torch.save(lenet.state_dict(), model_path)
        