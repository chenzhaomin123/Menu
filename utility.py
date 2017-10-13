import torch.autograd as autograd
from config import *
import numpy as np
def Variable(data, *args, **kwargs):
    if USE_CUDA:
        data = data.cuda()
    return autograd.Variable(data,*args, **kwargs)