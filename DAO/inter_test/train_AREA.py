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
from torch.autograd import Variable

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

def get_feature_mean(dataset_data_path, model, cls_num_list ):
    model.eval()
    cls_num=len(cls_num_list)
    feature_mean_end=torch.zeros(cls_num,256*15)
    print('feature mean processing...')
    with torch.no_grad():
        for name in tqdm(os.listdir(dataset_data_path),ncols=80):
            if int(name[4:8])<=52:
                input=np.load(dataset_data_path+name)[:,:,0::8]
                target=np.load(dataset_label_path+name)
                input=min_max_normalize(input)
                
                
                #idx=np.random.randint(0,input.shape[0],1)
                #input=input[idx]
                #target=target[idx]
                
                #print(input.shape,target.shape)
                input=torch.from_numpy(input).float().to(device)
                target=torch.from_numpy(target).long().to(device)
                
                input_var = to_var(input, requires_grad=False)
                output,features = model(input_var)
                
                #print(features.shape)#256,15
                features=features.detach()
                features = features.cpu().data.numpy()
                
                for out, label in zip(features, target):
                    
                    feature_mean_end[label]= feature_mean_end[label]+out.reshape(-1)
                
        img_num_list_tensor=torch.tensor(cls_num_list).unsqueeze(1)
        
        feature_mean_end=torch.div(feature_mean_end,img_num_list_tensor).detach() 
        return feature_mean_end



def calculate_eff_weight(dataset_data_path, model, cls_num_list,train_propertype ):
    model.eval()
    train_propertype=train_propertype.to(device)
    class_num=len(cls_num_list)
    eff_all= torch.zeros(class_num).float().to(device)
    print('eff weight processing...')
    with torch.no_grad():
        for name in tqdm(os.listdir(dataset_data_path),ncols=80):
            if int(name[4:8])<=52:
                input=np.load(dataset_data_path+name)[:,:,0::8]
                target=np.load(dataset_label_path+name)
                input=min_max_normalize(input)
                
                input=torch.from_numpy(input).float().to(device)
                target=torch.from_numpy(target).long().to(device)
                
                input_var = to_var(input, requires_grad=False)
                output,features = model(input_var)
                #print(features.shape)
                features=features.view(-1,256*15)
                
                mu=train_propertype[target].detach() #batch_size x d
                #print(mu.shape)
                feature_bz=(features.detach()-mu) #Centralization
                index = torch.unique(target) #class subset
                index2 = target.cpu().numpy()
                eff=torch.zeros(class_num).float().to(device)
            
                for i in range(len(index)): #number of class
                    index3 = torch.from_numpy(np.argwhere(index2==index[i].item()))
                    index3=torch.squeeze(index3)
                    feature_juzhen=feature_bz[index3].detach()
                    if  feature_juzhen.dim()==1:
                        eff[index[i]]=1
                    else:
                        _matrixA_matrixB = torch.matmul(feature_juzhen , feature_juzhen.transpose(0, 1))  
                        _matrixA_norm  = torch.unsqueeze(torch.sqrt(torch.mul(feature_juzhen, feature_juzhen).sum(axis=1)), 1) 
                        _matrixA_matrixB_length = torch.mul(_matrixA_norm , _matrixA_norm.transpose(0, 1))  
                        _matrixA_matrixB_length[_matrixA_matrixB_length==0]=1
                        r = torch.div(_matrixA_matrixB, _matrixA_matrixB_length).float() #R
                        num= feature_juzhen.size(0)
                        a=(torch.ones(1, num).float().to(device))/num #a_T
                        b=(torch.ones(num,1).float().to(device))/num #a
                        #print(r.type)
                        c=torch.matmul(torch.matmul(a,r),b).float().to(device) #a_T R a    
                        eff[index[i]]=1/c
                eff_all=eff_all+eff
                
        weights=eff_all
        weights = torch.where(weights>0, 1/weights, weights).detach()
        # weight
        fen_mu=torch.sum(weights)
        weights_new=weights/fen_mu
        weights_new=weights_new*class_num #Eq.(14)
        
        weights_new=weights_new.detach()
        
        return weights_new


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad)
    
if __name__=='__main__':

    dataset_data_path='../dataset/mass/data/'
    dataset_label_path='../dataset/mass/label/'
    
    torch.manual_seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    
    net=torch.load('./BASENET/save_model/40basenet_model.pth')
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    #loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    best_metrics=0.
    EPOCH=100
    
    
    print('load data...')
    #train_propertype,weights_old=None,None
    cls_num_list=np.zeros(5)
    for name in os.listdir(dataset_data_path):
        if int(name[4:8])<=52:
            label=np.load(dataset_label_path+name)
            for i in range(label.shape[0]):
                cls_num_list[int(label[i])]+=1
            
    for epoch in range(40,EPOCH):
        loss_all=[]
        
        train_propertype=get_feature_mean(dataset_data_path,net,cls_num_list )
        train_propertype=train_propertype.detach()
        
        weights_old=calculate_eff_weight(dataset_data_path, net, cls_num_list,train_propertype )
        weights_old=weights_old.detach()
        
        #print(weights_old)
    
        for name in tqdm(os.listdir(dataset_data_path),ncols=80):
            if int(name[4:8])<=52:
                data=np.load(dataset_data_path+name)[:,:,0::8]
                label=np.load(dataset_label_path+name)
                data=min_max_normalize(data)
                
                #idx=np.random.randint(0,data.shape[0],10)
                #data=data[idx]
                #label=label[idx]
                
                data=torch.from_numpy(data).float().to(device)
                label=torch.from_numpy(label).long().to(device).view(-1,1)
                
                output,_=net(data)
                loss=F.cross_entropy(output,label,weight=weights_old)
                
                loss_all.append(loss.detach().cpu().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print('epoch=%d,loss=%.3f'%(epoch,np.array(loss_all).mean()))
        if (epoch) % 10==0:
            precision,recall,f1,mcc,acc=test(net,device)
            if (precision+recall+f1+mcc+acc)>best_metrics:
                best_metrics=(precision+recall+f1+mcc+acc)
                torch.save(net,'./AREA/save_model/'+str(epoch)+'area_model.pth')
            f=open('./AREA/log/area_log.txt','a')
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

            
