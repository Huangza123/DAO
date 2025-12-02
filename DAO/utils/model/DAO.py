
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAO(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DAO, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv1d(self.n_channels,64,kernel_size=7,stride=2,padding=3) 
        self.pool1=nn.MaxPool1d(2)
        
        self.conv2 = nn.Sequential(
          nn.Conv1d(64,64,kernel_size=3,stride=1,padding=1,groups=32),
          
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.MaxPool1d(2),
        )
        self.identy_conv2=nn.Conv1d(64,64,kernel_size=5,stride=2,padding=2,groups=64)
        
        
        self.conv3 = nn.Sequential(
          nn.Conv1d(64,128,kernel_size=3,stride=1,padding=1,groups=32),
          
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.MaxPool1d(2),
        )
        #120
        self.identy_conv3=nn.Conv1d(64,128,kernel_size=5,stride=2,padding=2,groups=64)
        
        self.conv4 = nn.Sequential(
          nn.Conv1d(128,128,kernel_size=3,stride=1,padding=1,groups=64),
         
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.MaxPool1d(2),
        )#60
        self.identy_conv4=nn.Conv1d(128,128,kernel_size=5,stride=2,padding=2,groups=128)
        
        self.conv5 = nn.Sequential(
          nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1,groups=64),
         
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.MaxPool1d(2),
        )
        self.identy_conv5=nn.Conv1d(128,256,kernel_size=5,stride=2,padding=2,groups=128)
        
        
        self.conv6 = nn.Sequential(
          nn.Conv1d(256,256,kernel_size=3,stride=1,padding=1,groups=128),
         
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.MaxPool1d(2),
        )
        self.identy_conv6=nn.Conv1d(256,256,kernel_size=5,stride=2,padding=2,groups=256)
        
        self.out=nn.Conv1d(256,self.n_classes,kernel_size=1)
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        
        x2 = self.identy_conv2(x1)+self.conv2(x1)
        x3 = self.identy_conv3(x2)+self.conv3(x2)
        x4 = self.identy_conv4(x3)+self.conv4(x3)
        x5 = self.identy_conv5(x4)+self.conv5(x4)
        x6 = self.identy_conv6(x5)+self.conv6(x5)
        
        out=self.avgpool(self.out(x6))
        return out,x6
