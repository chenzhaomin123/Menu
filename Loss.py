import torch
import torch.nn as nn
from utility import Variable
from config import *
from CenterLoss import CenterLoss
class MyLoss(nn.Module):
    def __init__(self,num_classes, feat_dim, loss_weight=1.0):
        super(MyLoss,self).__init__()
        self.fc1 = nn.Linear(feat_dim,num_classes)
        self.center_loss = CenterLoss(num_classes, feat_dim, loss_weight)
        self.softmax_loss = nn.CrossEntropyLoss()
        if USE_CUDA:
            self.center_loss.cuda()
            self.softmax_loss.cuda()


    def forward(self,x,target):
        preb = self.fc1(x)

        pass
