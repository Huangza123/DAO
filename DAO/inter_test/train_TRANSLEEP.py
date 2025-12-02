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
from TRANSLEEP import *

def test(net,device):
    macro_avg_precision_set,macro_avg_recall_set,macro_avg_f1_set,_matthews_corrcoef_set,macro_avg_roc_auc_score_set,acc_set=[],[],[],[],[],[]
    
    for name in os.listdir(dataset_data_path):
        if int(name[4:8])>52:
            x0=np.load(dataset_data_path+name)[:,:,0::8]
            y0=np.load(dataset_label_path+name)
            x0=min_max_normalize(x0)
            #print(x0.shape)
            
            x=torch.from_numpy(x0).float().to(device).reshape(-1,1,27*960)
            y=torch.from_numpy(y0).long().view(-1)
            
            x = network.FE(x)
            x_att, l_1 = network.sce(x)
            h = network.bilstm(x_att)
            x = x.flatten(start_dim=2)
            
            #l_2_t = network.cls_st(h)
            #l_2_t = l_2_t.flatten(end_dim=1)
            
            h = network.dropout(network.project_f(x) + h)
            l_2 = network.cls(h)
            l_2 = l_2.flatten(end_dim=1)
                   
            predicts=torch.argmax(l_2,axis=1)
            predicts=predicts.cpu()
            
            macro_avg_f1=f1_score(y,predicts,average='macro',zero_division=True)
            macro_avg_precision=precision_score(y,predicts,average='macro',zero_division=True)
            macro_avg_recall=recall_score(y,predicts,average='macro',zero_division=True)
            _matthews_corrcoef=matthews_corrcoef(y,predicts)
            acc=accuracy_score(y,predicts)
                
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
        loss=torch.tensor([0.]).cuda(0)
        outputs=torch.softmax(outputs,axis=1)
        #print(outputs,'loss')
        for i in range(num_class):
          #print(labels==i)
          if len(outputs[labels==i])>0:
              #print(-torch.log(outputs[labels==i][:,i]),i,'loss')
              loss+=(-torch.log(outputs[labels==i][:,i]+1e-8).mean())
        return loss 
def loss_cross_entropy(weight=None, reduction='mean'):    # Take Logit as an Input
    return nn.CrossEntropyLoss(weight=weight, reduction=reduction)

def loss_cos_loss(margin=0, reduction='mean'):
    return nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

def loss_mse(reduction='none'):
    return torch.nn.MSELoss(reduction=reduction)

def loss_calculate(y_hat, y, y_pre=None, loss_type=None, ignore_index=None, regularizer_const=None, step=None):
    loss = loss_type(y_hat, y)
    return loss

def label_stage_transition(label):
    label_trans=np.zeros(label.shape)
    for i in range(0,label.shape[0]-1):
        if label[i-1] ==label[i] and label[i]==label[i+1]:
            label_trans[i]=0
        if label[i-1] !=label[i] and label[i]!=label[i+1]:
            label_trans[i]=1
    return label_trans    
    
if __name__=='__main__':

    dataset_data_path='../dataset/mass/data/'
    dataset_label_path='../dataset/mass/label/'
    
    torch.manual_seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)

    network=TranSleep(torch.rand(1024,1,27*960))#batch_size=512,input chnnels=27, feature_dimension=960
    
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    network.to(device)
    
    loss_func=ClassWiseEntropy()
    optimizer=torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    #batch_size=1024
    best_metrics=0.
    EPOCH=100
    
    
    print('load data...')
    train_data,train_label=[],[]
    lambda_sc,lambda_cls,lambda_cos_loss,lambda_st=2,1,2,0.2
    
    for epoch in range(EPOCH):
        loss_all=[]
        for name in tqdm(os.listdir(dataset_data_path),ncols=80):
            
            if int(name[4:8])<=52:
                x=np.load(dataset_data_path+name)[:,:,0::8].reshape(-1,1,27*960)
                y=np.load(dataset_label_path+name)
                y_t=label_stage_transition(y)
                
                x=min_max_normalize(x)
                
                #idx=np.random.randint(0,x.shape[0],512)
                #x=x[idx]
                #y=y[idx]
                #y_t=y_t[idx]
                
                x=torch.from_numpy(x).float().to(device)
                y=torch.from_numpy(y).long().to(device).view(-1)
                y_t=torch.from_numpy(y_t).long().to(device).view(-1)
                
                
                x = network.FE(x)
                x_att, l_1 = network.sce(x)
                h = network.bilstm(x_att)
                x = x.flatten(start_dim=2)
                l_2_t = network.cls_st(h)
                l_2_t = l_2_t.flatten(end_dim=1)
                
                #print(l_2_t.shape)
                h = network.dropout(network.project_f(x) + h)
                l_2 = network.cls(h)
                l_2 = l_2.flatten(end_dim=1)
        
        
                loss = 0
                l_1 = l_1.flatten(end_dim=1)
                
                loss_1 = lambda_sc*loss_func(l_1, y)
                loss = loss + loss_1
                
                loss_2 = lambda_cls*loss_func(l_2, y)
                loss_cos = lambda_cls*lambda_cos_loss*loss_cos_loss()(l_2, F.one_hot(y, num_classes=5), torch.ones_like(y))
                loss = loss + (loss_2 + loss_cos)
                
                loss_t = lambda_st*loss_func(l_2_t, y_t)
                loss = loss + loss_t
                
                loss_all.append(loss.detach().cpu().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #break
                
        print('epoch=%d,loss=%.3f'%(epoch,np.array(loss_all).mean()))
        if (epoch) % 10==0:
            with torch.no_grad():
                precision,recall,f1,mcc,acc=test(network,device)
                if (precision+recall+f1+mcc+acc)>best_metrics:
                    best_metrics=(precision+recall+f1+mcc+acc)
                    torch.save(network,'./TRANSLEEEP/save_model/'+str(epoch)+'transleep_model.pth')
                f=open('./TRANSLEEEP/log/transleep_log.txt','a')
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

                
