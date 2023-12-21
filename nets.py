import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict



class VGG(nn.Module):
    """
    Baseline VGG implementation for the  CIFAR10 DATASET
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 32x32-16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2 16x16-8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3 8x8-4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,10,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    

#model = VGG()
#print(model)


class VGG_BN(nn.Module):
    """
    Baseline VGG implementation for the  CIFAR10 DATASET with BN.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 32x32-16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2 16x16-8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3 8x8-4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,10,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    

#model2 = VGG_BN()
#print(model2)
    


class VGG_fullext(nn.Module):
    """
    Full extension of baseline VGG implementation for the  CIFAR10 DATASET
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 32x32-16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),                    # new
            nn.Conv2d(64,64,3,padding=1), # new
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2 16x16-8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),                      # new
            nn.Conv2d(128,128,3,padding=1), # new
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3 8x8-4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),                      # new
            nn.Conv2d(256,256,3,padding=1), # new
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,10,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    

class VGG_BN_fullext(nn.Module):
    """
    Full extension of baseline VGG implementation for the  CIFAR10 DATASET with BN.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 32x32-16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),           # new
            nn.ReLU(),                    # new
            nn.Conv2d(64,64,3,padding=1), # new
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2 16x16-8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),            # new
            nn.ReLU(),                      # new
            nn.Conv2d(128,128,3,padding=1), # new
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3 8x8-4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),            # new
            nn.ReLU(),                      # new
            nn.Conv2d(256,256,3,padding=1), # new
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500,10,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x



class VGG_toy(nn.Module):
    """
    Baseline VGG implementation with small dimensions for better testing.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 8x8-4x4
            nn.Conv2d(2, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2 4x4-2x2
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3 2x2-1x1
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

        )

        self.classifier = nn.Sequential(
            nn.Linear(16,5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5,2,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    

class VGG_BN_toy(nn.Module):
    """
    Baseline VGG implementation with small dimensions for better testing.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1 8x8-4x4
            nn.Conv2d(2, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2 4x4-2x2
            nn.Conv2d(4, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3 2x2-1x1
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

        )

        self.classifier = nn.Sequential(
            nn.Linear(16,5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5,2,bias=False)
        )

        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


def init_vgg(model_baseline):

    return model_baseline




def build_vgg_baseline(BN=False):
    if BN:
        model= VGG_BN()
        model = init_vgg(model)
    else: 
        model = VGG()
        model = init_vgg(model)
    return model
    
def build_vgg_fullyext(BN=False):
    if BN:
        freezed_numbers=[2,3,4,5,10,11,12,13,18,19,20,21]
        freezed=[]
        model = VGG_BN_fullext()
        for i,p in enumerate(model.parameters()):
            if i in freezed_numbers:
                freezed.append(p)
        return model, freezed
    else: 
        freezed_numbers=[2,3,6,7,10,11]
        freezed = []
        model = VGG_fullext()
        for i,p in enumerate(model.parameters()):
            if i in freezed_numbers:
                freezed.append(p)
        return model, freezed


def extend_VGG(position, BN=False):
    '''
    generates model which is extended by one layer.

    Args:
        position: either 0,1,2 dep on the position in the vgg baseline
        BN: indicates whether batch normalization is used in the architecture
    '''
    if position not in [0,1,2]:
        print(F'Error: {position} is not feasible!')
        return 0
    
    class VGG_ext(nn.Module):
        def __init__(self,BN=False, position=0):
            super().__init__()

            modules = []

            modules.append(nn.Conv2d(3, 64, 3, padding=1))
            if position==0:
                if BN:
                    modules.append(nn.BatchNorm2d(64))
                modules.append(nn.ReLU())
                modules.append(nn.Conv2d(64,64,3,padding=1))
            if BN:
                modules.append(nn.BatchNorm2d(64))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2, stride=2, return_indices=True))


            modules.append(nn.Conv2d(64, 128, 3, padding=1))
            if position==1:
                if BN:
                    modules.append(nn.BatchNorm2d(128))
                modules.append(nn.ReLU())
                modules.append(nn.Conv2d(128,128,3,padding=1))
            if BN:
                modules.append(nn.BatchNorm2d(128))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2, stride=2, return_indices=True))

            modules.append(nn.Conv2d(128, 256, 3, padding=1))
            if position==2:
                if BN:
                    modules.append(nn.BatchNorm2d(256))
                modules.append(nn.ReLU())
                modules.append(nn.Conv2d(256,256,3,padding=1))
            if BN:
                modules.append(nn.BatchNorm2d(256))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2, stride=2, return_indices=True))

            self.features = nn.Sequential(*modules)

            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500,10,bias=False)
            )

            
        def forward(self, x):
            for layer in self.features:
                if isinstance(layer, nn.MaxPool2d):
                    x, location = layer(x)
                else:
                    x = layer(x)
            
            x = x.view(x.size()[0], -1)
            x = self.classifier(x)
            return x
    
    model = VGG_ext(BN, position)
    return model


#model = extend_VGG(2,BN=True)
#print(model)
#for i,p in enumerate(model.parameters()):
#    print(i)
#    print(p.shape)

