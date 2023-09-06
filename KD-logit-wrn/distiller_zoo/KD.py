from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss



if __name__ == '__main__':
    import torch
    
    x = torch.randn(64,512,4,4)
    y = torch.randn(64,512,4,4)
    x = torch.randn(64,64,32,32)
    y = torch.randn(64,64,32,32)
    x = torch.randn(64,100)
    y = torch.randn(64,100)
    kd = DistillKL(T=4)
    
    loss = kd(x,y)
    print(loss)