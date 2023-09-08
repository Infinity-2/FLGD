from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

class MY_KD(nn.Module):

    def __init__(self, p=2):
        super(MY_KD, self).__init__()
        self.conv1 = nn.Conv2d(16,128,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,128,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128,128,kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU()
        # self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, 100)
        
        # self.conv1.requires_grad = False
        self.conv1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.conv4.weight.requires_grad = False

        self.p = p

    def forward(self, g_s, g_t):
        # print(g_s.shape)
        for f_s, f_t in zip(g_s, g_t):
            # print(f_s.shape)
            # print(f_t.shape)
            if f_s.shape[1] == 16:
                f_s, f_t = self.fun1(f_s, f_t)
            if f_s.shape[1] == 32:
                f_s, f_t = self.fun2(f_s, f_t)
            if f_s.shape[1] == 64:
                f_s, f_t = self.fun3(f_s, f_t)
            if f_s.shape[1] == 128:
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
    
    def MY_loss(self, f_s, f_t):
        # Huber_loss = nn.SmoothL1Loss()
        # return(Huber_loss(f_s,f_t))
        return ((f_t - f_s).pow(2).mean())
        
    
    def fun1(self, f_s, f_t):
        f_s = self.conv1(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        f_s = F.avg_pool2d(f_s,8)
        # f_s = self.pool(f_s)
        f_s = f_s.view(f_s.size(0), -1)
        f_s = self.classifier(f_s)
        
        f_t = self.conv1(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        f_t = F.avg_pool2d(f_t,8)
        # f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        return f_s, f_t
        
    def fun2(self, f_s, f_t):
        f_s = self.conv2(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        f_s = F.avg_pool2d(f_s,8)
        f_s = f_s.view(f_s.size(0), -1)
        f_s = self.classifier(f_s)
        
        f_t = self.conv2(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        f_t = F.avg_pool2d(f_t,8)
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        return f_s, f_t
        
    def fun3(self, f_s, f_t):
        import torch
        device = torch.device('cuda')
        f_s = self.conv3(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        f_s = F.avg_pool2d(f_s,8)
        f_s = f_s.view(f_s.size(0), -1)
        f_s1 = nn.Linear(f_s.size(1),100)
        f_s1 = f_s1.to(device)
        f_s = f_s1(f_s)
        
        f_t = self.conv3(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        f_t = F.avg_pool2d(f_t,8)
        f_t = f_t.view(f_t.size(0), -1)
        f_t1 = nn.Linear(f_t.size(1),100)
        f_t1 = f_s1.to(device)
        f_t = f_t1(f_t)
        
        return f_s, f_t
    
    
    def fun4(self, f_s, f_t):
        f_s = self.conv4(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        f_s = F.avg_pool2d(f_s,8)
        f_s = f_s.view(f_s.size(0), -1)
        f_s = self.classifier(f_s)
        
        f_t = self.conv4(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        f_t = F.avg_pool2d(f_t,8)
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        return f_s, f_t
        
        
if __name__ == '__main__':
    import torch
    
    x = torch.randn(64,16,32,32)
    y = torch.randn(64,16,32,32)
    
    l = MY_KD()
    x,y = l(x,y)

    print(x.shape,y.shape)
