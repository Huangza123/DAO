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
from DASNN import *

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
            # outputs_roc=torch.softmax(outputs,axis=1)
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
        loss=torch.tensor([0.]).cuda(1)
        outputs=torch.softmax(outputs,axis=1)
        #print(outputs,'loss')
        for i in range(num_class):
          #print(labels==i)
          if len(outputs[labels==i])>0:
              #print(-torch.log(outputs[labels==i][:,i]),i,'loss')
              loss+=(-torch.log(outputs[labels==i][:,i]+1e-8).mean())
        return loss 

if __name__=='__main__':

    dataset_data_path='../dataset/mass/data/'
    dataset_label_path='../dataset/mass/label/'
    
    torch.manual_seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    
    #net=DASNN(27,5)#input chnnels, output classes.
    net=torch.load('./BASENET/save_model/40basenet_model.pth')
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    #loss_func=nn.CrossEntropyLoss()
    loss_func=ClassWiseEntropy()
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    #batch_size=1024
    best_metrics=0.
    EPOCH=100
    
    
    print('load data...')
    train_data,train_label=[],[]
    for epoch in range(40,EPOCH):
        loss_all=[]
        for name in tqdm(os.listdir(dataset_data_path),ncols=80):
            if int(name[4:8])<=52:
                data=np.load(dataset_data_path+name)[:,:,0::8]
                label=np.load(dataset_label_path+name)
                data=min_max_normalize(data)
                
                #idx=np.random.randint(0,data.shape[0],256)
                #data=data[idx]
                #label=label[idx]
                
                data=torch.from_numpy(data).float().to(device)
                label=torch.from_numpy(label).float().to(device).view(-1)
                
                output,_=net(data)
                loss=loss_func(output,label)
                
                loss_all.append(loss.detach().cpu().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print('epoch=%d,loss=%.3f'%(epoch,np.array(loss_all).mean()))
        if (epoch) % 10==0:
            precision,recall,f1,mcc,acc=test(net,device)
            if (precision+recall+f1+mcc+acc)>best_metrics:
                best_metrics=(precision+recall+f1+mcc+acc)
                torch.save(net,'./BASENET/save_model/'+str(epoch)+'basenet_model.pth')
            f=open('./BASENET/log/basenet_log.txt','a')
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

            
