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
from torch.autograd import Variable
from torch import autograd
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
        for i in range(num_class):
          #print(labels==i)
          if len(outputs[labels==i])>=0:
              #print(-torch.log(outputs[labels==i][:,i]),i,'loss')
              loss+=(-torch.log(outputs[labels==i][:,i]+1e-8).mean())
        return loss 
def ewc_loss(net, lamda, cuda=True):
    try:
        losses = []
        for n, p in net.named_parameters():
            # retrieve the consolidated mean and fisher information.
            
            n = n.replace('.', '__')
            mean = getattr(net, '{}_estimated_mean'.format(n))
            fisher = getattr(net, '{}_estimated_fisher'.format(n))
            # wrap mean and fisher in variables.
            mean = Variable(mean)
            fisher = Variable(fisher)
            # calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((fisher * (p-mean)**2).sum())
        return (lamda/2)*sum(losses)
    except AttributeError:
        # ewc loss is 0 if there's no consolidated parameters.
        return (
            Variable(torch.zeros(1)).cuda(1) if cuda else
            Variable(torch.zeros(1))
        )
def consolidate(net, fisher):
    for n, p in net.named_parameters():
        n = n.replace('.', '__')
        net.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
        net.register_buffer('{}_estimated_fisher'.format(n), fisher[n].data.clone())

def is_on_cuda(net):
    return next(net.parameters()).is_cuda
                
def estimate_fisher(net, task_index, sample_size=500, batch_size=32):
    # sample loglikelihoods from the dataset.
    loglikelihoods = []
    #for x, y in data_loader:
    for name in tqdm(os.listdir(dataset_data_path),ncols=80):
                
        if int(name[4:8]) in task_index:
            data=np.load(dataset_data_path+name)[:,:,0::8]
            label=np.load(dataset_label_path+name)
            data=min_max_normalize(data)
            
            idx=np.random.randint(0,data.shape[0],batch_size)
            data=data[idx]
            label=label[idx]
            
            x=torch.from_numpy(data)
            y=torch.from_numpy(label).view(-1)
            
            #x = x.view(batch_size, -1)
            x,y=x.float(),y.float()
            x = Variable(x).cuda(1) if is_on_cuda(net) else Variable(x)
            y = Variable(y).cuda(1) if is_on_cuda(net) else Variable(y)
            
            loglikelihoods.append(
                F.log_softmax(net(x)[0].reshape(batch_size,-1),dim=1)[range(0,batch_size), y.data.long()]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
    # estimate the fisher information of the parameters.
    loglikelihood = torch.cat(loglikelihoods).mean(0)
    loglikelihood_grads = autograd.grad(loglikelihood, net.parameters())
    parameter_names = [
        n.replace('.', '__') for n, p in net.named_parameters()
    ]
    return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}
        
if __name__=='__main__':
    dataset_data_path='../dataset/mass/data/'
    dataset_label_path='../dataset/mass/label/'
    
    torch.manual_seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    
    net=DASNN(27,5)#input chnnels, output classes.
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    #loss_func=nn.CrossEntropyLoss()
    loss_func=ClassWiseEntropy()
    
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    #batch_size=1024
    
    EPOCH=100
    
    print('load data...')
    
    task_index=[[],[],[],[]]
    for i in range(52):
        task_index[i//13].append(i+1)
               
            
    train_epoch_per_task=100
    best_epoch=0
    for tsk in range(len(task_index)):
        best_metrics=0.
        best_metrics_all=[]
            
        for epoch in range(train_epoch_per_task):                               
            for name in tqdm(os.listdir(dataset_data_path),ncols=80):
                
                if int(name[4:8]) in task_index[tsk]:
                    data=np.load(dataset_data_path+name)[:,:,0::8]
                    label=np.load(dataset_label_path+name)
                    
                    data=min_max_normalize(data)
                    for i in range(2):
                        idx=np.random.randint(0,data.shape[0],512)
                        data=data[idx]
                        label=label[idx]
                        
                        inputs=torch.from_numpy(data).to(device).float()
                        labels=torch.from_numpy(label).to(device).view(-1)
                                
                        #inputs,labels=data
                        inputs,labels=inputs.float().to(device),labels.float().to(device)
                        output,_=net(inputs)
                        
                        loss=loss_func(output,labels,5)
                        if tsk>=1:
                            ewcloss=ewc_loss(net,5e+3)
                            loss=loss+ewcloss    
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                
            if epoch%10==0:
                
                precision,recall,f1,mcc,acc=test(net,device)
                if (precision+recall+f1+mcc+acc)>best_metrics:
                    best_metrics=(precision+recall+f1+mcc+acc)
                    best_metrics_all=[precision,recall,f1,mcc,acc]
                    best_epoch=epoch
                    torch.save(net,'./EWC/save_model/'+str(tsk)+'_'+str(epoch)+'ewc_model.pth')
                f=open('./EWC/log/ewc_log.txt','a')
                f.write(str(['[%d,%d]'%(tsk,epoch),
                  'precision=%.4f'%precision,
                  'recall=%.4f'%recall,
                  'f1=%.4f'%f1,
                  'mcc=%.4f'%mcc,
                  'acc=%.4f'%acc]))      
                f.write('\n')
                f.close()
                print('[%d,%d]'%(tsk,epoch),
                  'precision=%.4f'%precision,
                  'recall=%.4f'%recall,
                  'f1=%.4f'%f1,
                  'mcc=%.4f'%mcc,
                  'acc=%.4f'%acc
                  ) 
                  
          
        consolidate(net,estimate_fisher(net,task_index[tsk], sample_size=512, batch_size=512))
        
        f=open('./EWC/log/ewc_log.txt','a')
        f.write(str(['task=%d '%(tsk),'best metrics:',str(best_metrics_all)]))      
        f.write('\n')
        f.close()
              
