# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:12:55 2025

@author: Lenovo
"""

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import os
from sklearn.metrics import precision_score, recall_score, f1_score,matthews_corrcoef,roc_auc_score,accuracy_score
from  tqdm import *
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.functional as F
import sys
sys.path.append('../utils/model/')
from DAO import *

class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M

        self.LWS = LWS
        self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()

        self.BN_H = nn.BatchNorm1d(feat_in).cuda()
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`

    """

    def __init__(self, eps, max_iter, dis, gpu, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dis = dis
        self.gpu = gpu

    def forward(self, x, y):
        
        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8).cuda()
        x_col = x.unsqueeze(-2).cuda()
        y_lin = y.unsqueeze(-3).cuda()
        
        if self.dis == 'cos':
            C = 1-d_cosine(x_col, y_lin)
        elif self.dis == 'euc':
            C= torch.mean((torch.abs(x_col - y_lin)) ** 2, -1)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()

        actual_nits = 0
        thresh = 1e-1

        for i in range(self.max_iter):
            u1 = u  
            u = self.eps * (torch.log(mu+1e-8).cuda() - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8).cuda() - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def _cost_matrix(x, y, dis, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == 'cos':
            C = 1-d_cosine(x_col, y_lin)
        elif dis == 'euc':
            C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)

        return C



    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
        
def test(net,device):
    macro_avg_precision_set,macro_avg_recall_set,macro_avg_f1_set,_matthews_corrcoef_set,macro_avg_roc_auc_score_set,acc_set=[],[],[],[],[],[]
    for name in os.listdir(dataset_data_path):
        if int(name[4:8])>52:
            data=np.load(dataset_data_path+name)[:,:,0::8]
            label=np.load(dataset_label_path+name)
            data=min_max_normalize(data)
        
            data=torch.from_numpy(data).float().to(device)
            label=torch.from_numpy(label).float().to(device)
            
            outputs,_=net(data)
            outputs_roc=torch.softmax(outputs,axis=1)
            # print(outputs_roc)
            predicts=torch.argmax(outputs,axis=1)
            labels,predicts=label.cpu(),predicts.cpu()
            
            macro_avg_f1=f1_score(labels,predicts,average='macro',zero_division=True)
            macro_avg_precision=precision_score(labels,predicts,average='macro',zero_division=True)
            macro_avg_recall=recall_score(labels,predicts,average='macro',zero_division=True)
            _matthews_corrcoef=matthews_corrcoef(labels,predicts)
            acc=accuracy_score(labels,predicts)
            
            acc_set.append(acc)
            macro_avg_precision_set.append(macro_avg_precision)
            macro_avg_recall_set.append(macro_avg_recall)
            macro_avg_f1_set.append(macro_avg_f1)
            _matthews_corrcoef_set.append(_matthews_corrcoef)
    return np.array(macro_avg_precision_set).mean(),np.array(macro_avg_recall_set).mean(),np.array(macro_avg_f1_set).mean(),np.array(_matthews_corrcoef_set).mean(),np.array(acc_set).mean()
def min_max_normalize(data):
    for i in range(data.shape[0]):
        min_value,max_value=np.min(data[i]),np.max(data[i])
        data[i]=(data[i]-min_value)/(max_value-min_value)   
    return data 
class ClassWiseEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,outputs,labels,num_class=5):
        loss=torch.tensor([0.]).cuda()
        outputs=torch.softmax(outputs,axis=1)
        #print(outputs,'loss')
        for i in range(num_class):
          #print(labels==i)
          if len(outputs[labels==i])>0:
              #print(-torch.log(outputs[labels==i][:,i]),i,'loss')
              loss+=(-torch.log(outputs[labels==i][:,i]+1e-8).mean())
        return loss        
def etf_loss(data,label,net,classifier_eft):        
    inputs=torch.from_numpy(data).float().to(device)
    labels=torch.from_numpy(label).float().to(device).view(-1)
    
    #output=net(data)
    
    batch_size = inputs.size()[0]

    index = torch.randperm(batch_size).to(device)
    #lam = np.random.beta(args.alpha, args.alpha)
    lam = np.random.beta(0.1, 0.1)
    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
    
    y_a, y_b = labels, labels[index]
    
    #labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).long()
    # print(labels[0],labels_one_hot[0])
    #print(mixed_x.shape,'mxshape')
    outputs,feature=net(mixed_x)
    
    #outputs=torch.softmax(outputs,dim=1)
    #print(inputs.shape,'outputs')
    feature=feature.reshape(inputs.shape[0],256*15).to(device)

    feat_etf=classifier_etf(feature)    
    # print(feat_etf.shape,'fea',y_a,y_b)
    y_a,y_b=y_a.reshape(-1),y_b.reshape(-1)
    loss_ot = sinkhorna_ot(feat_etf, classifier_etf.ori_M.T)
    #loss2=0.
    #if epoch >50:
    """data2=np.load('./buffer/data/'+name)
    label2=np.load('./buffer/label/'+name)
    data2=min_max_normalize(data2)
    
    inputs2=torch.from_numpy(data2).float().to(device)
    labels2=torch.from_numpy(label2).long().view(-1).to(device)
    
    outputs2,feature2=net(inputs2)
    
    loss2 = loss_func(outputs2, labels2.view(-1).long())
    """    
    loss = loss_func(outputs, y_a.view(-1,1).long()) * lam + loss_func(outputs, y_b.view(-1,1).long()) * (1. - lam) + 0.1 * loss_ot
    return loss
if __name__=='__main__':
    
    dataset_data_path='../dataset/mass/data/'
    dataset_label_path='../dataset/mass/label/'
    
    torch.manual_seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    
    net=DAO(27,5)
    net=torch.load('./DISA/save_model/50disa_model.pth')
   
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    #net.to(device)
    
    
    classifier_etf=ETF_Classifier(256*15,5)
    sinkhorna_ot = SinkhornDistance(eps=1, max_iter=200, reduction=None, dis='cos', gpu=0)

    #print(net)
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    #batch_size=1024
    best_metrics=0.
    EPOCH=200
    
    
    print('synthetic adversarial minority data...')
    train_data,train_label=[],[]
    for name in tqdm(os.listdir(dataset_data_path),ncols=80):
        if int(name[4:8])<=52:
            data=np.load(dataset_data_path+name)[:,:,0::8]
            label=np.load(dataset_label_path+name)
            num=np.zeros(5)
            for i in range(label.shape[0]):
                num[int(label[i])]+=1
            
            max_idx=np.argmax(num)
            
            data=min_max_normalize(data)
            
            inputs=torch.from_numpy(data[label!=max_idx]).float().to(device)
            labels=torch.from_numpy(label[label!=max_idx]).long().to(device)
            
            delta=torch.zeros((inputs.size()),requires_grad=True)
            
            goptimizer=torch.optim.Adam([delta], lr=0.03, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
            gloss_func=nn.CrossEntropyLoss()
            #gloss_all=[]
            save=False
            
            ginputs=inputs+delta.to(device)
            for gepoch in range(20):
                goutput,gfeature=net(ginputs)
                goutput=torch.softmax(goutput,dim=1)
                gloss=torch.exp(-gloss_func(goutput,labels.view(-1,1)))
                gprediction=torch.argmax(goutput,axis=1)
                
                goptimizer.zero_grad()
                gloss.backward()
                goptimizer.step()
                ginputs=inputs+delta.to(device)
                
            np.save('./DAO/buffer/data/'+name,ginputs.detach().cpu().numpy())
            np.save('./DAO/buffer/label/'+name,labels.detach().cpu().numpy())
    print('start training...')      
    net=torch.nn.DataParallel(net,device_ids=[0])
          
    for epoch in range(50,EPOCH):
        loss_all=[]
        
        for name in tqdm(os.listdir(dataset_data_path),ncols=80):
            if int(name[4:8])<=52:
                data=np.load(dataset_data_path+name)[:,:,0::8]
                label=np.load(dataset_label_path+name)
                data=min_max_normalize(data)
                
                loss_ori=etf_loss(data,label,net,classifier_etf)
                
                data2=np.load('./DAO/buffer/data/'+name)
                label2=np.load('./DAO/buffer/label/'+name)
                data2=min_max_normalize(data2)
                loss_gen=etf_loss(data2,label2,net,classifier_etf)
                
                
                loss=loss_ori+0.1*loss_gen
                
                loss_all.append(loss.detach().cpu().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                    
        
        print('epoch=%d,loss=%.3f'%(epoch,np.array(loss_all).mean()))
        if (epoch) % 10==0:
            precision,recall,f1,mcc,acc=test(net,device)
            if (precision+recall+f1+mcc+acc)>best_metrics:
                best_metrics=(precision+recall+f1+mcc+acc)
                torch.save(net,'./DAO/save_model/'+str(epoch)+'dao_model.pth')
            f=open('./DADAOSNN/dao_log.txt','a')
            f.write(str(['[%d,%d]'%(epoch,EPOCH),
              'precision=%.4f'%precision,
              'recall=%.4f'%recall,
              'f1=%.4f'%f1,
              'mcc=%.4f'%mcc,
              'acc=%.4f'%acc]))      
            f.write('\n')
            f.close()
            print('[%d,%d]'%(epoch,EPOCH),
              'precision=%.4f'%precision,
              'recall=%.4f'%recall,
              'f1=%.4f'%f1,
              'mcc=%.4f'%mcc,
              'acc=%.4f'%acc
              )
              
              
              
              