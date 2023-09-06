from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class MY_KD(nn.Module):

    def __init__(self, p=2):
        super(MY_KD, self).__init__()
        self.conv1 = nn.Conv2d(16,128,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,128,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16,128,kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32,128,kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128,128,kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64,128,kernel_size=3, padding=1)

        self.bn1   = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU()
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(8192, 100)
        self.lin1 = nn.Linear(131072, 100)
        self.lin2 = nn.Linear(131072, 100)
        self.lin3 = nn.Linear(32768, 100)
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

    def forward(self, g_s, g_t, lt):
        return [self.MY_loss(f_s,f_t,lt) for f_s, f_t in zip(g_s, g_t)]








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
    
    def MY_loss(self, f_s, f_t, lt):
        import torch
        gram_logit = torch.mm(lt, torch.t(lt))
        gram_logit = F.normalize(gram_logit)
        
        if f_t.shape[1] == 16:
            # print(f_s.shape)
            f_s1, f_t1 = self.fun1(f_s, f_t)
            # print('fun')
        if f_t.shape[1] == 32:
            f_s1, f_t1 = self.fun2(f_s, f_t)
        if f_t.shape[1] == 64:
            f_s1, f_t1 = self.fun3(f_s, f_t)
        if f_t.shape[1] == 128:
            f_s1, f_t1 = self.fun4(f_s, f_t)
        # return(((f_s - f_t)**2).mean())
        loss_stand = (gram_logit - f_s1)
        G_diff = (f_t1 - f_s1)
        return(((G_diff * G_diff).sum() / 64)*10 + (loss_stand ** 2).sum() / 64)
        # p_s = F.log_softmax(f_s/4, dim=1)
        # p_t = F.softmax(f_t/4, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (4**2) / f_s.shape[0]
        # return loss
        # Huber_loss = nn.SmoothL1Loss()
        # loss = nn.MSELoss()
        
        
    
    def fun1(self, f_s, f_t):
        import torch
        f_s = self.conv1(f_s)
        f_s = self.bn1(f_s)
        # print(f_s.shape)
        f_s = self.relu(f_s)
        # print(f_s.shape)
        fs1 =f_s
        # f_s = self.pool(f_s)
        # print(f_s.shape)
        f_s = f_s.view(f_s.size(0), -1 )
        # print(f_s.shape)
        # lin= nn.Linear(524288, 100)
        f_s = self.lin1(f_s)
        # f_s = self.classifier(f_s)
        # f_s = F.softmax(f_s / 4, dim=1)
        f_t = self.conv1(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        ft1 = f_t
        # f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        # f_t = self.classifier(f_t)
        # lin= nn.Linear(524288, 100)
        f_t = self.lin1(f_t)
        # f_t = F.softmax(f_t / 4, dim=1)
        pred_s = f_s
        pred_t = f_t
        GramM_s = torch.mm(pred_s, torch.t(pred_s))
        GramM_t = torch.mm(pred_t, torch.t(pred_t))
        
        #归一化去掉了 
        GramM_s = F.normalize(GramM_s)
        GramM_t = F.normalize(GramM_t)
        # huber_loss = torch.nn.SmoothL1Loss()
        # G_diff = (GramM_s-GramM_t)
        # loss = (G_diff * G_diff).view(-1, 1).sum() 
        # print('LOSS')
        # print(loss)
        f_s = GramM_s
        f_t = GramM_t
        # loss1 = torch.nn.MSELoss()
        # print(loss1(f_t, f_s))
        # print(((f_s - f_t)**2).mean())
        
        # print(((f_s - f_t)**2).sum()/64)
        return f_s, f_t
        
    def fun2(self, f_s, f_t):
        import torch
        f_s = self.conv3(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        # print(f_s.shape)
        # f_s = self.pool(f_s)
        f_s = f_s.view(f_s.size(0), -1)
        # print(f_s.shape)
        # f_s = self.classifier(f_s)
        f_s = self.lin2(f_s)
        
        f_t = self.conv2(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        # f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        # f_t = self.classifier(f_t)
        f_t = self.lin2(f_t)
        
        
        pred_s = f_s
        pred_t = f_t
        GramM_s = torch.mm(pred_s, torch.t(pred_s))
        GramM_t = torch.mm(pred_t, torch.t(pred_t))
        GramM_s = F.normalize(GramM_s)
        GramM_t = F.normalize(GramM_t)
        # huber_loss = torch.nn.SmoothL1Loss()
        # G_diff = (GramM_s-GramM_t)
        # loss = (G_diff * G_diff).view(-1, 1).sum() 
        # print('LOSS')
        # print(loss)
        f_s = GramM_s
        f_t = GramM_t
        # print(((f_s - f_t)**2).mean())
        
        # print(((f_s - f_t)**2).sum()/64)
        return f_s, f_t
        
    def fun3(self, f_s, f_t):
        import torch
        f_s = self.conv5(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        # f_s = self.pool(f_s)
        f_s = f_s.view(f_s.size(0), -1)
        # print(f_s.shape)
        # f_s = self.classifier(f_s)
        f_s = self.lin3(f_s)
        
        f_t = self.conv4(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        # f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        # f_t = self.classifier(f_t)
        f_t = self.lin3(f_t)
        
        pred_s = f_s
        pred_t = f_t
        huber_loss = torch.nn.SmoothL1Loss()
        # print(huber_loss(pred_s,pred_t))
        
        # pred_s = F.softmax(f_s / 4 , dim=1)
        # pred_t = F.softmax(f_t / 4, dim=1)

        GramM_s = torch.mm(pred_s, torch.t(pred_s))
        GramM_t = torch.mm(pred_t, torch.t(pred_t))
        # print(huber_loss(GramM_s,GramM_t))
        GramM_s = F.normalize(GramM_s)
        GramM_t = F.normalize(GramM_t)
        # print(huber_loss(GramM_s,GramM_t))
        huber_loss = torch.nn.SmoothL1Loss()
        G_diff = (GramM_s-GramM_t)
        loss = (G_diff * G_diff).view(-1, 1).sum() 
        # print('LOSS')
        # print(loss)
        f_s = GramM_s
        f_t = GramM_t
        
        # print(((f_s - f_t)**2).sum()/64)
        return f_s, f_t
    
    
    def fun4(self, f_s, f_t):
        import torch
        f_s = self.conv7(f_s)
        f_s = self.bn1(f_s)
        f_s = self.relu(f_s)
        # print(f_s.shape)
        # f_s = self.pool(f_s)
        f_s = f_s.view(f_s.size(0), -1)
        # print(f_s.shape)
        f_s = self.classifier(f_s)
        
        f_t = self.conv6(f_t)
        f_t = self.bn1(f_t)
        f_t = self.relu(f_t)
        # f_t = self.pool(f_t)
        f_t = f_t.view(f_t.size(0), -1)
        f_t = self.classifier(f_t)
        pred_s = f_s
        pred_t = f_t
        # pred_s = F.softmax(f_s , dim=1)
        # pred_t = F.softmax(f_t , dim=1)
        GramM_s = torch.mm(pred_s, torch.t(pred_s))
        GramM_t = torch.mm(pred_t, torch.t(pred_t))
        GramM_s = F.normalize(GramM_s)
        GramM_t = F.normalize(GramM_t)
        huber_loss = torch.nn.SmoothL1Loss()
        G_diff = (GramM_s-GramM_t)
        loss = (G_diff * G_diff).view(-1, 1).sum() 
        # print('LOSS')
        # print(loss)
        f_s = GramM_s
        f_t = GramM_t
        
        # print(((f_s - f_t)**2).sum()/64)
        return f_s, f_t
        
    
if __name__ == '__main__':
    import torch
    
    x = torch.randn(64,64,8,8)
    y = torch.randn(64,128,8,8)
    fun = MY_KD()
    
    x1, x2 = fun.fun4(x,y)
    
    # loss = torch.nn.MSELoss()
    # print(loss(x,y))
    # # print(loss(x3,x4))
    # print(loss(x1,x2))
    
    # print(x.shape,y.shape)
