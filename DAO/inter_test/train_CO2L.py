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
import copy

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
        loss=torch.tensor([0.]).cuda(0)
        outputs=torch.softmax(outputs,axis=1)
        #print(outputs,'loss')
        for i in range(num_class):
          #print(labels==i)
          if len(outputs[labels==i])>0:
              #print(-torch.log(outputs[labels==i][:,i]),i,'loss')
              loss+=(-torch.log(outputs[labels==i][:,i]+1e-8).mean())
        return loss 

def augment(data,num=2):
    aug_data0_set,aug_data1_set=[],[]
    for j in range(data.shape[0]):
        seed=data[j]
        dis=np.sum(np.sum((data[j]-data)**2,axis=1),axis=1)
        dis[np.argmin(dis)]=np.max(dis)
        min_dis=np.argmin(dis)
        aug_data0=seed+(data[min_dis]-seed)*np.random.uniform(0,1)
        aug_data1=seed+(data[min_dis]-seed)*np.random.uniform(0,1)
        aug_data0_set.append(aug_data0)
        aug_data1_set.append(aug_data1)
    aug_data0=np.array(aug_data0_set).reshape(data.shape)
    aug_data1=np.array(aug_data1_set).reshape(data.shape)
    return aug_data0,aug_data1
    
if __name__=='__main__':
    dataset_data_path='../dataset/mass/data/'
    dataset_label_path='../dataset/mass/label/'
    
    torch.manual_seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    
    net=DASNN(27,5)#input chnnels, output classes.
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    #net = nn.DataParallel(net)
    net.to(device)
    
    
    loss_func=ClassWiseEntropy()
    
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    EPOCH=100
    
    print('load data...')
    train_data_stream=[[],[],[]]
    train_label_stream=[[],[],[]]    
    load_count=0  
    
    #TASK DATA
    task_index=[[],[],[],[]]
    for i in range(52):
        task_index[i//13].append(i+1)
      
    train_epoch_per_task=100
    
    best_metrics=0.
    best_metrics_all=[]                          
                    
    for tsk in range(len(task_index)):
        best_metrics=0.
        best_metrics_all=[]
        
        memory_buffer_data=[]
        memory_buffer_label=[]
        print('task %d training...'%tsk)
        if tsk==0:
            print('task %d memory buffer update...'%tsk)
            for name in tqdm(os.listdir(dataset_data_path),ncols=80):
                if int(name[4:8]) in task_index[0]:  #13 person, each person raondomly select 200//13=15 instances
                    data=np.load(dataset_data_path+name)[:,:,0::8]
                    label=np.load(dataset_label_path+name)
                    for c in range(5):
                        class_c_data=data[label==c]
                        ridx=np.random.randint(0,class_c_data.shape[0],5)
                        memory_buffer_data.append(class_c_data[ridx])
                        memory_buffer_label.append(np.ones(5)*c)
        else:
            print('task %d memory buffer update...'%tsk)
            for rtsk in range(tsk):
                for name in tqdm(os.listdir(dataset_data_path),ncols=80):
                    if int(name[4:8]) in task_index[rtsk]:
                        data=np.load(dataset_data_path+name)[:,:,0::8]
                        label=np.load(dataset_label_path+name)
                        for c in range(5):
                            class_c_data=data[label==c]
                            if len(class_c_data)>0:
                                ridx=np.random.randint(0,class_c_data.shape[0],5)
                                memory_buffer_data.append(class_c_data[ridx])
                                memory_buffer_label.append(np.ones(5)*c)
                            else:
                                ridx=np.random.randint(0,data.shape[0],5)
                                memory_buffer_data.append(data[ridx])
                                memory_buffer_label.append(label[ridx])
        
        memory_buffer_data=np.array(memory_buffer_data).reshape(-1,27,960)
        memory_buffer_label=np.array(memory_buffer_label).reshape(-1)
        ridx=np.random.randint(0,memory_buffer_data.shape[0],200)
        
        memory_buffer_data=memory_buffer_data[ridx]
        memory_buffer_label=memory_buffer_label[ridx]
        
        memory_buffer_data_tensor=torch.from_numpy(memory_buffer_data).to(device).float()
        memory_buffer_label_tensor=torch.from_numpy(memory_buffer_label).to(device)
        
        if tsk>=1:
            net_old=copy.deepcopy(net)
            
            for epoch in range(train_epoch_per_task):
                mean_loss=0.
                for name in tqdm(os.listdir(dataset_data_path),ncols=80):
                    loss_ce,loss_supcon,loss_ird=0.,0.,0.
                    if int(name[4:8]) in task_index[tsk]:
                        
                        data=np.load(dataset_data_path+name)[:,:,0::8]
                        label=np.load(dataset_label_path+name)
                        data=min_max_normalize(data)
                        
                        ridx=np.random.randint(0,data.shape[0],256)
                        data=data[ridx]
                        label=label[ridx]
                        
                        #augmented batch size with 
                        aug_data0,aug_data1=augment(data,2)
                        
                        aug_data=np.concatenate((aug_data0,aug_data1),axis=0)
                        aug_label=np.concatenate((label,label),axis=0)
                        
                        tensor_data=torch.from_numpy(aug_data).to(device).float()
                        tensor_label=torch.from_numpy(aug_label).to(device).view(-1)
                        
                        output,feature=net(tensor_data)
                        output_old,feature_old=net_old(tensor_data)
                        
                        #print(feature.shape,feature_old.shape)
                        loss_ce0=loss_func(output,tensor_label)
                        
                        mem_buffer_output,mem_buffer_feature=net(memory_buffer_data_tensor)
                        mem_buffer_output_old,mem_buffer_feature_old=net_old(memory_buffer_data_tensor)
                        #print(mem_buffer_feature.shape,mem_buffer_feature_old.shape)
                        
                        loss_ce1=loss_func(mem_buffer_output,memory_buffer_label_tensor.view(-1).long())
                        loss_ce=loss_ce0+loss_ce1
                        
                        mem_buffer_feature=mem_buffer_feature.view(mem_buffer_feature.shape[0],-1)
                        mem_buffer_feature_old=mem_buffer_feature_old.view(mem_buffer_feature_old.shape[0],-1)
                        
                        mean,var=torch.mean(feature),torch.var(feature)
                        mean_old,var_old=torch.mean(feature_old),torch.var(feature_old)
                        
                        feature=((feature-mean)/var)
                        mem_buffer_feature=((mem_buffer_feature-mean)/var)
                        
                        feature_old=((feature_old-mean_old)/var_old)
                        mem_buffer_feature_old=((mem_buffer_feature_old-mean_old)/var_old)
                        
                        #print(feature,mem_buffer_feature)
                        # only current task samples as anchols
                        anchols_feature=feature.view(feature.shape[0], -1) #batch_size,channels,-1
                        anchols_feature_old=feature.view(feature_old.shape[0], -1) #batch_size,channels,-1
                        
                        # the same class of the current task as positives, past task from the memory buffer as negatives. 
                        #contrast_feature=anchols_feature
                        loss_supcon,loss_ird=0.,0.
                        for i in range(anchols_feature.shape[0]):
                            anchols=anchols_feature[i]
                            
                            positive_features=anchols_feature[aug_label==aug_label[i]]
                            
                            all_sum=torch.exp(torch.sum(anchols*anchols_feature,axis=1)/7e4)
                            all_sum=all_sum.sum()-all_sum[i]
                            all_sum+=(torch.exp(torch.sum(anchols*mem_buffer_feature,axis=1)/7e4).sum())
                            
                            anchol_constrast=torch.exp(torch.sum(anchols*positive_features,axis=1)/7e4)
                            loss_supcon+=((-1/anchols_feature.shape[0])*torch.log(anchol_constrast/all_sum).sum())
                            
                            p_ij=anchol_constrast/all_sum
                            
                            #ird loss 
                            all_sum_old=torch.exp(torch.sum(anchols_feature_old[i]*anchols_feature_old,axis=1)/7e4)
                            all_sum_old=all_sum_old.sum()-all_sum_old[i]
                            
                            #print(anchols_feature_old.shape,mem_buffer_feature_old.shape,anchols.shape,anchols_feature.shape)
                            
                            all_sum_old+=(torch.exp(torch.sum(anchols_feature_old[i]*mem_buffer_feature_old,axis=1)/7e4).sum())
                            
                            anchol_constrast_old=torch.exp(torch.sum(anchols_feature_old[i]*anchols_feature_old[aug_label==aug_label[i]],axis=1)/7e4)
                            
                            p_ij_old=anchol_constrast_old/all_sum_old
                            
                            
                            loss_ird+=(-p_ij_old*torch.log(p_ij)).sum()
                            #print(tsk,loss_ird,loss_supcon)
                        loss=loss_ce+0.01*loss_supcon+0.01*loss_ird
                        mean_loss+=loss   
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                print('tsk=%d, epoch=%d, loss=%.3f'%(tsk,epoch,mean_loss/13))
                if epoch%10==0:
                    precision,recall,f1,mcc,acc=test(net,device)
                    if (precision+recall+f1+mcc+acc)>best_metrics:
                        best_metrics=(precision+recall+f1+mcc+acc)
                        best_metrics_all=[precision,recall,f1,mcc,acc]
                        best_epoch=epoch
                        torch.save(net,'./CO2L/save_model/tsk_'+str(tsk)+'_'+str(epoch)+'co2l_model.pth')
                    f=open('./CO2L/log/co2l_log.txt','a')
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
            f=open('./CO2L/log/co2l_log.txt','a')
            f.write(str(['task=%d '%(tsk),'best metrics:',str(best_metrics_all)]))      
            f.write('\n')
            f.close()
        else:       
            #loss_func=nn.CrossEntropyLoss()
            for epoch in range(train_epoch_per_task):
                mean_loss=0.
                for name in tqdm(os.listdir(dataset_data_path),ncols=80):
                    loss_ce,loss_supcon,loss_ird=0.,0.,0.
                    if int(name[4:8]) in task_index[tsk]:
                        
                        data=np.load(dataset_data_path+name)[:,:,0::8]
                        label=np.load(dataset_label_path+name)
                        data=min_max_normalize(data)
                        #print(data.shape,'data_shape')
                        ridx=np.random.randint(0,data.shape[0],256)
                        data=data[ridx]
                        label=label[ridx]
                        
                        #augmented batch size with 
                        aug_data0,aug_data1=augment(data,2)
                        
                        aug_data=np.concatenate((aug_data0,aug_data1),axis=0)
                        aug_label=np.concatenate((label,label),axis=0)
                        
                        tensor_data=torch.from_numpy(aug_data).to(device).float()
                        tensor_label=torch.from_numpy(aug_label).to(device).view(-1)
                        
                        output,feature=net(tensor_data)
                        #output_old,feature_old=net_old(tensor_data)
                        
                        #print(output.shape,aug_label.shape)
                        loss_ce0=loss_func(output,tensor_label)
                        
                        mem_buffer_output,mem_buffer_feature=net(memory_buffer_data_tensor)
                        #mem_buffer_output_old,mem_buffer_feature_old=net_old(memory_buffer_data_tensor)
                        
                        loss_ce1=loss_func(mem_buffer_output,memory_buffer_label_tensor.view(-1).long())
                        
                        loss_ce=loss_ce0+loss_ce1
                        mem_buffer_feature=mem_buffer_feature.view(mem_buffer_feature.shape[0],-1)
                        
                        mean,var=torch.mean(feature),torch.var(feature)
                        
                        feature=((feature-mean)/var)
                        mem_buffer_feature=((mem_buffer_feature-mean)/var)
                        
                        #print(feature,mem_buffer_feature)
                        # only current task samples as anchols
                        anchols_feature=feature.view(feature.shape[0], -1) #batch_size,channels,-1
                        #anchols_feature_old=feature.view(feature_old.shape[0], -1) #batch_size,channels,-1
                        
                        # the same class of the current task as positives, past task from the memory buffer as negatives. 
                        #contrast_feature=anchols_feature
                        
                        for i in range(anchols_feature.shape[0]):
                            anchols=anchols_feature[i]
                            
                            positive_features=anchols_feature[aug_label==aug_label[i]]
                            
                            all_sum=torch.exp(torch.sum(anchols*anchols_feature,axis=1)/7e4)
                            all_sum=all_sum.sum()-all_sum[i]
                            all_sum+=(torch.exp(torch.sum(anchols*mem_buffer_feature,axis=1)/7e4).sum())
                            
                            anchol_constrast=torch.exp(torch.sum(anchols*positive_features,axis=1)/7e4)
                            loss_supcon+=((-1/anchols_feature.shape[0])*torch.log(anchol_constrast/all_sum).sum())
                            
                            #p_ij=anchol_constrast/all_sum
                            
                            #ird loss 
                            #all_sum_old=torch.exp(torch.sum(anchols_features_old[i]*anchols_features_old,axis=1)/7e4)
                            #all_sum_old=all_sum_old.sum()-all_sum_old[i]
                            #all_sum_old+=(torch.exp(torch.sum(anchols_features_old[i]*mem_buffer_feature_old,axis=1)/7e4).sum())
                            
                            #anchol_constrast_old=torch.exp(torch.sum(anchols*positive_features,axis=1)/7e4)
                            
                            #p_ij_old=anchol_constrast_old/all_sum_old
                            
                            #loss_ird+=(-p_ij_old*torch.log(p_ij))
                            #print(tsk,loss_ird,loss_supcon)
                            
                        loss=loss_ce+0.01*loss_supcon+0.01*loss_ird
                        mean_loss+=loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        #break
                print('tsk=%d, epoch=%d, loss=%.3f'%(tsk,epoch,mean_loss/13))       
                if epoch%10==0:
                    precision,recall,f1,mcc,acc=test(net,device)
                    if (precision+recall+f1+mcc+acc)>best_metrics:
                        best_metrics=(precision+recall+f1+mcc+acc)
                        best_metrics_all=[precision,recall,f1,mcc,acc]
                        best_epoch=epoch
                        torch.save(net,'./CO2L/save_model/tsk_'+str(tsk)+'_'+str(epoch)+'co2l_model.pth')
                    f=open('./CO2L/log/co2l_log.txt','a')
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
            f=open('./CO2L/log/co2l_log.txt','a')
            f.write(str(['task=%d '%(tsk),'best metrics:',str(best_metrics_all)]))      
            f.write('\n')
            f.close()
              
        

