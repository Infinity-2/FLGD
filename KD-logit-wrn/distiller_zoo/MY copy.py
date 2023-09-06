from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class MY_KD(nn.Module):

    def __init__(self, p=2):
        super(MY_KD, self).__init__()
        self.conv1 = nn.Conv2d(64,512,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128,512,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256,512,kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512,512,kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512,512,kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(512)
        self.relu  = nn.ReLU()
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(8192, 100)
        # import torch
        # torch.nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

        # torch.nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')

        # torch.nn.init.kaiming_uniform_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        
        # self.conv1.weight.requires_grad = False
        # self.conv2.weight.requires_grad = False
        # self.conv3.weight.requires_grad = False
        # self.conv4.weight.requires_grad = False
        # self.classifier.weight.requires_grad = False
        # self.conv1.weight.to('cuda')
        # self.conv2.weight.to('cuda')
        # self.conv3.weight.to('cuda')
        # self.conv4.weight.to('cuda')
        # self.classifier.weight.to('cuda')
        
        
        # self.pool.weight.requires_grad = False

        self.p = p

    def forward(self, g_s, g_t):
        for f_s, f_t in zip(g_s, g_t):
            # print(len(g_s))

            # print(f_s.shape)
            # print(f_t.shape)
            if f_s.shape[1] == 64:
                f_s, f_t = self.fun1(f_s, f_t)
                # print('fun')
            if f_s.shape[1] == 128:
                f_s, f_t = self.fun2(f_s, f_t)
            if f_s.shape[1] == 256:
                f_s, f_t = self.fun3(f_s, f_t)
            if f_s.shape[1] == 512:
                f_s, f_t = self.fun4(f_s, f_t)
        return [self.MY_loss(f_s,f_t)]








    # def at_loss(self, f_s, f_t):
    #     s_H, t_H = f_s.shape[2], f_t.shape[2]
    #     if s_H > t_H:
    #         f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    #     elif s_H < t_H:
    #         f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    #     else:
    #         pass
    #     return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    # def at(self, f):
    #     return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
    
    
        # p_s = F.log_softmax(y_s/self.T, dim=1)
        # p_t = F.softmax(y_t/self.T, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        # return loss
    
    def MY_loss(self, f_s, f_t):
        # p_s = F.log_softmax(f_s/4, dim=1)
        # p_t = F.softmax(f_t/4, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (4**2) / f_s.shape[0]
        # return loss
        Huber_loss = nn.SmoothL1Loss()
        loss = nn.MSELoss()
        return(Huber_loss(f_s,f_t))
        
    
    def fun1(self, f_s, f_t):
        f_s = self.conv1(f_s)
        
        f_s = self.bn1(f_s)
        
        f_s = self.relu(f_s)
        print(f_s.shape)
        fs1 =f_s
        f_s = self.pool(f_s)
        print(f_s.shape)
        
        f_s = f_s.view(f_s.size(0), -1 )
        print(f_s.shape)
        f_s = self.classifier(f_s)
        
        f_t = self.conv1(f_t)
        
        f_t = self.bn1(f_t)
        
        f_t = self.relu(f_t)
        ft1 = f_t
        f_t = self.pool(f_t)
        
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        
        loss1 = torch.nn.MSELoss()
        print(loss1(fs1, ft1))
        
        return f_s, f_t
        
    def fun2(self, f_s, f_t):
        f_s = self.conv2(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        f_s = self.pool(f_s)
        f_s = f_s.view(f_s.size(0), -1)
        f_s = self.classifier(f_s)
        
        f_t = self.conv2(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        return f_s, f_t
        
    def fun3(self, f_s, f_t):
        f_s = self.conv3(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        f_s = self.pool(f_s)
        f_s = f_s.view(f_s.size(0), -1)
        f_s = self.classifier(f_s)
        
        f_t = self.conv3(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        
        return f_s, f_t
    
    
    def fun4(self, f_s, f_t):
        f_s = self.conv4(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        print(f_s.shape)
        # f_s = self.pool(f_s)
        f_s = f_s.view(f_s.size(0), -1)
        print(f_s.shape)
        f_s = self.classifier(f_s)
        
        f_t = self.conv4(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        # f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        return f_s, f_t
        
        
if __name__ == '__main__':
    import torch
    
    x = torch.randn(64,512,4,4)
    y = torch.randn(64,512,4,4)
    fun = MY_KD()
    
    x1, x2 = fun.fun4(x,y)
    
    loss = torch.nn.MSELoss()
    print(loss(x,y))
    # print(loss(x3,x4))
    print(loss(x1,x2))
    
    print(x.shape,y.shape)
