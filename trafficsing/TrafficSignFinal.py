#importing libraries
import glob
import os
import os.path
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import codecs
import errno
import torchvision.datasets
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import  pathlib


transforms=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])
])


test_path='./TestImages'
train_path='./TrainingDataset'

trainloader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transforms),
    batch_size=256,shuffle=True
)

testloader=DataLoader(torchvision.datasets.ImageFolder(test_path,transform=transforms),
                      batch_size=256,shuffle=True)


root=pathlib.Path(train_path)
classes=sorted([i.name.split('/')[-1]] for i in root.iterdir())




class ConvNet(nn.Module):
    def __init__(self,num_classes=20):
        super(ConvNet,self).__init__()

        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #shape=(256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()

        self.pool=nn.MaxPool2d(kernel_size=2)

        #reduce image size ( 256,12,75,75)

        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #(256,20,75,75)
        self.relu2=nn.ReLU()

        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #(256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()

        self.fc1=nn.Linear(32*75*75,out_features=num_classes)


    def forward(self,input):

        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)

        output=self.pool(output)


        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3()

        #above the output will be in matrix form, with shape (256,32,75,75)

        output=output.view(-1,32,75*75)
        output=self.fc1(output)

        return output



device=torch.device('cuda' if torch.cuda.is_available() else 'cput')

model=ConvNet(num_classes=20).to(device)


#optimizer and loss function

optimizer=Adam(model.parameters(),lr=.001,weight_decay=.0001)
loss_function=nn.CrossEntropyLoss()


train_count=len(glob.glob(train_path+'/**/*.jpg'))

test_count=len(glob.glob(test_path+'/**/*.jpg'))




