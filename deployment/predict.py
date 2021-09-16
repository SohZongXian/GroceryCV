#Load libraries
import time
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt

import cv2 
from PIL import Image 
import copy


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


train_path='C:/Users/user/Documents/GitHub/GroceryCV/GroceryStoreDataset-master/dataset_pt/train'
train_path2='C:/Users/user/Documents/GitHub/GroceryCV/GroceryStoreDataset-master/dataset/train'

root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
root2=pathlib.Path(train_path2)
classes2=sorted([i.name.split('/')[-1] for i in root2.iterdir()])

# print(classes)
# print(classes2)

class ConvNet(nn.Module):
    def __init__(self,num_classes=len(classes)):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
        
        
        #Feed foward function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (256,32,75,75)
            
        output=output.view(-1,32*75*75)
            
            
        output=self.fc(output)
            
        return output



## model predict 

img_file_path = '../sample_images/natural/Banana.jpg'

#Transforms
transformer_infer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

## loading models
m = torch.load('grocery_lessclass.pth')
m.eval()
n = torch.load('grocery_moreclass.pth')
n.eval()

def prediction1(img_path, transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        m.cuda()
    
    input = Variable(image_tensor)
    output=m(input.to(device))
    
    index = output.cpu().data.numpy().argmax()
    pred = classes[index]
    return pred


def prediction2(img_path, transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        n.cuda()
    
    input = Variable(image_tensor)
    output=n(input.to(device))
    
    index = output.cpu().data.numpy().argmax()
    pred = classes2[index]
    return pred



print(prediction1(img_file_path, transformer_infer))
print(prediction2(img_file_path, transformer_infer))

