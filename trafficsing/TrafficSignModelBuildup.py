import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
#from torchsummary import summary
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data_transofrm=transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor()
])

BATCH_SIZE=256
learning_rate=0.0001
EPOCHS=15
numClasses=43


train_path='myData'
train_data=torchvision.datasets.ImageFolder(root=train_path,transform=data_transofrm)

ratio=.8
total_input_data=len(train_data)
n_train_sample=int(len(train_data)*ratio)
n_validation_sample=total_input_data-n_train_sample
print(n_validation_sample)

#now print

print('Todal input data: ',total_input_data)
print('Total train datasets: ',n_train_sample)
print('Total Validation datasets :' ,n_validation_sample)

#randomly separated train and validation data

train_data,validation_data=data.random_split(train_data,[n_train_sample,n_validation_sample])
print('Total Validation datasets :' ,len(validation_data))
#Plot and histogram of the training and validation datasets

train_hist=[0]*numClasses
for i in train_data.indices:
    tar=train_data.dataset.targets[i]
    train_hist[tar]+=1

val_hist=[0]*numClasses
for i in validation_data.indices:
    tar=validation_data.dataset.targets[i]
    val_hist[tar]+=1
plt.bar(range(numClasses),train_hist,label='train')
plt.bar(range(numClasses),val_hist,label='validation')
lagend=plt.legend(loc='upper right',shadow=True)
plt.title('Distribution plot')
plt.xlabel('Class ID')
plt.ylabel('Number of examples')
plt.savefig('train_vali_split.png',bbox_inches='tight',pad_inches=.05)
plt.show()


train_loader = data.DataLoader(train_data, shuffle=True, batch_size = BATCH_SIZE)
val_loader = data.DataLoader(validation_data, shuffle=True, batch_size = BATCH_SIZE)

def count_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from TrafficSignModel import  AlexnetTS

model=AlexnetTS(numClasses)
print(f'Model has {count_parameter(model):,}trainable parameter')


# import torchsummary
# print(summary(model, (3, 112, 112)))