from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class MY_KD(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(MY_KD, self).__init__()
        
    # def forward(self, g_s, g_t):
    #     return [self.loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def forward(self, y_s, y_t):
        
        # y_s = y_s.cpu()
        # y_t = y_t.cpu()
        # y_s = y_s.numpy()
        # y_t = y_t.detach().numpy()
        # GramM_s = np.dot(y_s.T, y_s)
        # GramM_t = np.dot(y_t.T, y_t)
        # GramM_s = torch.from_numpy(GramM_s)
        # GramM_t = torch.from_numpy(GramM_t)
        # Vs, Ss = torch.symeig(GramM_s, eigenvectors=True)
        # Vt, St = torch.symeig(GramM_t, eigenvectors=True)
        # Vs = F.normalize(Vs, dim = 1)
        
        
        
# #----------------------------*-------------------------------------

#可以用的
#         y_s = y_s.cpu()
#         y_t = y_t.cpu()
#         y_s = y_s.detach().numpy()
#         y_t = y_t.detach().numpy()
#         GramM_s = np.dot(y_s.T, y_s)
#         GramM_t = np.dot(y_t.T, y_t)
#         norms = np.linalg.norm(GramM_s, axis=1, keepdims=True)
#         normt = np.linalg.norm(GramM_t, axis=1, keepdims=True)
#         GramM_s = GramM_s / norms
#         GramM_t = GramM_t / normt
        
#         U_s, S_s, V_s = np.linalg.svd(GramM_s)
        
#         U_t, S_t, V_t = np.linalg.svd(GramM_t)
        
#         U_s = torch.from_numpy(U_s)
#         S_s = torch.from_numpy(S_s)
#         V_s = torch.from_numpy(V_s)
        
        
#         U_t = torch.from_numpy(U_t)
#         S_t = torch.from_numpy(S_t)
#         V_t = torch.from_numpy(V_t)
#         U_s = U_s.cuda()
#         S_s = S_s.cuda()
#         V_s = V_s.cuda()
        
#         U_t = U_t.cuda()
#         S_t = S_t.cuda()
#         V_t = V_t.cuda()
# #----------------------------*-------------------------------------
        # pred_s = F.softmax(y_s / 4, dim = 1)
        # pred_t = F.softmax(y_t.detach() / 4, dim = 1)
        
# #------------------------------------*-------------------------------
# #格拉姆矩阵可以训练的代码，精度73.6
        pred_s = y_s
        pred_t = y_t
        GramM_s = torch.mm(pred_s, torch.t(pred_s))
        GramM_t = torch.mm(pred_t, torch.t(pred_t))
        GramM_s = F.normalize(GramM_s)
        GramM_t = F.normalize(GramM_t)
        huber_loss = torch.nn.SmoothL1Loss()
        G_diff = (GramM_s-GramM_t)
        loss = (G_diff * G_diff).view(-1, 1).sum() / (64 * 64)
        return [loss]
# #-----------------------------------*--------------------------------
        
        
#-----------------------------------------*----------------------------------------
#精度为67.98
        # pred_s = y_s
        # pred_t = y_t
        # GramM_s = torch.mm(pred_s, torch.t(pred_s))
        # GramM_t = torch.mm(pred_t, torch.t(pred_t))
        # GramM_s = F.normalize(GramM_s)
        # GramM_t = F.normalize(GramM_t)
        
        # su, sv = torch.symeig(GramM_s, eigenvectors=True)
        # tu, tv = torch.symeig(GramM_t, eigenvectors=True)
        # Eigen_diff = (tu - su)
        # # G_diff = (GramM_s-GramM_t)
        # # loss = (G_diff * G_diff).view(-1, 1).sum() / (64 * 64)
        # loss = (Eigen_diff * Eigen_diff).view(-1, 1).sum() / (64 * 64)
        # return [loss]
#-----------------------------------------*----------------------------------------

    
if __name__ == '__main__':
    import torch
    
    x = torch.randn(64,100)
    y = torch.randn(64,100)
    fun = MY_KD(T = 2)
    X1,X2,X3 = fun(x,y)

    print(X1,X2,X3)
