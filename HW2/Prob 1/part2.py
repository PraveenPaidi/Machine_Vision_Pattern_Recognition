#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision.transforms import RandomRotation
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import RandomAffine
from torch.utils.data import ConcatDataset
import random


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=random.choice([0, 20,50, 0, 0, 0])),
    transforms.RandomHorizontalFlip(p=random.choice([0, 0.2,0.5, 0, 0, 0])),
    transforms.RandomVerticalFlip(p=random.choice([0, 0.2,0.5, 0, 0, 0])),
    transforms.RandomAffine(degrees=(random.choice([0, 20,50, 0, 0, 0]))),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 6

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


trainloader_aug  = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[4]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))


# In[5]:


for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader_aug, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
                
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


# In[6]:


total = 0 
correct =0

with torch.no_grad():
    
    for data in testloader:
    
        inputs, labels=data

        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# In[ ]:





# In[ ]:





# In[ ]:






# transform1 = transforms.Compose([transforms.ToTensor(),
#     RandomRotation(degrees=15),
# ])

# transform2 = transforms.Compose([transforms.ToTensor(),
#     RandomHorizontalFlip(p=0.5),
# ])

# transform3 = transforms.Compose([transforms.ToTensor(),
#     RandomVerticalFlip(p =0.5), 
# ])

# transform4 = transforms.Compose([transforms.ToTensor(),
#     RandomAffine((30,70)),
# ])


# trainset_aug1 = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=(transform1))

# trainset_aug2 = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=(transform2))

# trainset_aug3 = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=(transform3))

# trainset_aug4 = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=(transform4))


# combined_dataset = ConcatDataset([trainset, trainset_aug1, trainset_aug2, trainset_aug3, trainset_aug4])

