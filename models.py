'''
Created on 1 Sep 2021

@author: aliv
'''

import metrics
import pickle
import time
import os
import numpy as np
from scipy import spatial
from scipy.spatial.distance import cosine
import json
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from data_util import read_pkl, kfold_train
from tqdm import tqdm
from losses import *



def compute_EB(Grads):
    m = Grads.mean(axis = 0)
    s = Grads.std(axis = 0)
    s[s==0] = 0.001
    return 1. - Grads.shape[0]*np.square(m/s).mean()

def compute_GD(Grads, s=5):
    s = min(s,Grads.shape[0])
    batch_size = int(Grads.shape[0]/s)
    batched_grads = []
    for i in range(s):
        batched_grads.append(Grads[i*batch_size:(i+1)*batch_size,:].mean(axis = 0))
    avg_grad_dis = 0
    cnt2 = 0
    for i in range(s):
        for j in range(s):
            if i < j:
                grads1 = batched_grads[i]
                grads2 = batched_grads[j]
                GD = np.linalg.norm(grads1-grads2)
                avg_grad_dis += GD
                cnt2 += 1
    # to avoid division by zero here, args.s must be >= 2
    avg_grad_dis /= cnt2
    
    return avg_grad_dis
    


def evaluate(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.tefm, dataset.tedlr)
    if saveto is not None:
        results = {}
        results['dlr'] = dataset.tedlr
        results['lv'] = dataset.telv
        results['pred'] = y_pred
        with open(saveto, 'wb') as f:
            pickle.dump(results, 
                        f)
    metric = metrics.LTRMetrics(dataset.telv,np.diff(dataset.tedlr),y_pred)

    return metric.NDCG(1), metric.NDCG(3), metric.NDCG(5), metric.NDCG(10), y_pred#, metric.NDCG_perquery(1)

def evaluate_valid(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.vafm, dataset.vadlr)
    if saveto is not None:
        results = {}
        results['dlr'] = dataset.vadlr
        results['lv'] = dataset.valv
        results['pred'] = y_pred
        with open(saveto, 'wb') as f:
            pickle.dump(results,
                        f)

    metric = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),y_pred)

    return metric.NDCG(1), metric.NDCG(3), metric.NDCG(5), metric.NDCG(10)

def evaluate_train(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.trfm, dataset.trdlr)
    if saveto is not None:
        results = {}
        results['dlr'] = dataset.trdlr
        results['lv'] = dataset.trlv
        results['pred'] = y_pred
        with open(saveto, 'wb') as f:
            pickle.dump(results,
                        f)

    metric = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),y_pred)
    return metric.NDCG(1), metric.NDCG(3), metric.NDCG(5), metric.NDCG(10)


def predict_dnn(net):
  
    if torch.cuda.is_available():
        torch_ = torch.cuda
    else:
        torch_ = torch
    def predict(fm, dlr):
        dl = DataLoader(QueryGroupedLTRData(fm, dlr, np.ones(fm.shape[0])), batch_size=1, shuffle=False)
        y = []
        with torch.no_grad():
          for (x, _) in dl:
            output = net(x)[0,:,:]
#             print(output.shape)
            y.append(np.mean(output.cpu().data.numpy(), 1))
        return np.concatenate(y)
    return predict



class LTRData(Dataset):
    def __init__(self, allfm, alllv):
        dev = get_torch_device()
        
        if torch.cuda.is_available():
          self.torch_ = torch.cuda
        else:
          self.torch_ = torch
          
        self.features = self.torch_.FloatTensor(allfm, device=dev)
        self.labels = self.torch_.FloatTensor(alllv, device=dev)
    
    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]
    
class QueryGroupedLTRData(Dataset):
    def __init__(self, fm, dlr, lv):
        self.feature = fm
        self.dlr = dlr
        self.labels = lv
        self.dev = get_torch_device()
        if torch.cuda.is_available():
            self.torch_ = torch.cuda
        else:
            self.torch_ = torch
    def __len__(self):
        return self.dlr.shape[0] - 1
    def __getitem__(self, qid):
        s_i = self.dlr[qid]
        e_i = self.dlr[qid+1]
        feature = self.torch_.FloatTensor(self.feature[s_i:e_i,:], device=self.dev)
        labels = self.torch_.FloatTensor(self.labels[s_i:e_i], device=self.dev)
        return feature, labels


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=None):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_layers[0], bias=False)]
        for i in range(1, len(hidden_layers)):
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_layers[-1], 1, bias=False))
        
        self.layers = nn.Sequential(*layers)
        
    def override_first_layers(self, net):
        with torch.no_grad():
            for i in range(len(net.w_all)-1):
                self.w_all[i][:] = net.w_all[i]
    
    def forward(self, x):
        return self.layers(x)
    
    
    def save(self, path_to_model):
        torch.save(self.state_dict(), path_to_model)
      
    def train_batch(self, loss_fn, x, y):
        self.opt.zero_grad()
        out = self(x)#[:,0]
        loss = loss_fn(out, y)
        loss.backward()
        self.opt.step()
            
        grads1_s = []
        
        for param in self.w:
            grads1_s.append(param.grad.view(-1))
            grads_last = param.grad.view(-1)
        grads1_s = torch.cat(grads1_s)
        return grads1_s.data.cpu().numpy()[None,:], grads_last.data.cpu().numpy()[None,:]
      
    
    def params(self, only_last=False):
        self.w_all = list(self.parameters())
        if only_last:
            self.w = self.w_all[-1:]
        else:
            self.w = self.w_all
        return self.w
    
def set_seed(random_seed):
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed) 


def predict_linear(net, feature_map_fn):
  
    if torch.cuda.is_available():
        torch_ = torch.cuda
    else:
        torch_ = torch
    def predict(fm, dlr):
        dl = DataLoader(QueryGroupedLTRData(fm, dlr, np.ones(fm.shape[0])), batch_size=1, shuffle=False)
        y = []
        with torch.no_grad():
            for (x, _) in dl:
                output = net(feature_map_fn(x))[0,:,:]
    #             print(output.shape)
                y.append(np.mean(output.cpu().data.numpy(), 1))
        return np.concatenate(y)
    return predict

def log_epoch(epoch, epochs):
    if epoch < 0:
        return False
    if epoch == epochs - 1 or epoch < 25:
        return True
    steps = [[100,5],[500,10],[1000,20],[10000,50]]
    for step, div in steps:
        if epoch < step and (epoch+1) % div == 0:
            return True
    return False


def train_model(jobid, dataset, dataset_name, 
                 layers,
                 epochs, learning_rate, l2,
                 rseed,
                 optimizer_str, loss_fn_str,
                 first_layers_train_epochs,
                 cv_folds,
                 results_file):
    results_file = os.path.join(results_file, jobid + '.json')
    
    folds = kfold_train(dataset, cv_folds)
    cv_train_dls = [DataLoader(QueryGroupedLTRData(folds[i].trfm, folds[i].trdlr, folds[i].trlv), batch_size=1, shuffle=True) for i in range(len(folds))]
    train_dl = DataLoader(QueryGroupedLTRData(dataset.trfm, dataset.trdlr, dataset.trlv), batch_size=1, shuffle=True)
    
    start_time = int(time.time())
    
    
    loss_fn = eval(loss_fn_str)

    opt_fn = eval(f'torch.optim.{optimizer_str}')
        
        
    cv_nets = []
    for i in range(cv_folds):
        set_seed(rseed)
        cv_nets.append(DNN(dataset.trfm.shape[1], layers))
        cv_nets[-1].opt = opt_fn(cv_nets[-1].params(first_layers_train_epochs==0), lr=learning_rate, weight_decay=l2)
    
    set_seed(rseed)
    net = DNN(dataset.trfm.shape[1], layers)
    net.opt = opt_fn(net.params(first_layers_train_epochs==0), lr=learning_rate, weight_decay=l2)
    
    cdr_nets = []
    cv_running = True
    
    if torch.cuda.is_available():
        device = get_torch_device()
        
        net.cuda(device)
        for i in range(len(cv_nets)):
            cv_nets[i].cuda(device)

    validation_ndcg = []
    alphas = [np.ones(5) for i in range(2)]
    
    for epoch in range(epochs):
        
        batch = 0
        ndcgs = []
        net.train()
        Grads, Grads_last = [], []
        for (x, y) in train_dl:
            grads_s, grads_last = net.train_batch(loss_fn, x, y)
            Grads.append(grads_s)
            Grads_last.append(grads_last)
        Grads = np.concatenate(Grads, axis=0)
        Grads_last = np.concatenate(Grads_last, axis=0)
        
        if epoch >= first_layers_train_epochs:
            if not cdr_nets:
                for i in range(2):
                    cdr_nets.append(DNN(dataset.trfm.shape[1], layers))
                    
                    if torch.cuda.is_available():
                        cdr_nets[-1].cuda(get_torch_device())
            
                    cdr_nets[-1].opt = opt_fn(cdr_nets[-1].params(True), lr=learning_rate, weight_decay=l2)
                    cdr_nets[-1].override_first_layers(net)
            
            for i in range(len(cdr_nets)):
                cdr_nets[i].train()
                for (x, y) in train_dl:
                    cdr_nets[i].train_batch(loss_fn, x, y)
                alphas[i] = cdr_nets[i].w[-1].data.cpu().numpy()
            
        if cv_running:
            for i in range(len(cv_nets)):
                cv_nets[i].train()
                for (x, y) in cv_train_dls[i]:
                    cv_nets[i].train_batch(loss_fn, x, y)

        results = {}
        cdr_eval = 0
        if log_epoch(epoch, epochs) or log_epoch(epoch-first_layers_train_epochs, epochs):
            tmp = evaluate(dataset, predict_dnn(net.eval()))
            results['test'] = tmp[:-1]
#             results['train'] = evaluate_train(dataset, predict_dnn(net.eval()))
            
            if epoch >= first_layers_train_epochs:
                cdr_eval = evaluate(dataset, predict_dnn(cdr_nets[0].eval()))[-2]
            
            if cv_running:
                ndcgs = []
                for i in range(len(cv_nets)):
                    ndcgs.append(evaluate_valid(folds[i], predict_dnn(cv_nets[i].eval()))[-1])
                validation_ndcg.append(np.array(ndcgs).mean())
                
                if len(validation_ndcg) > 7:
                    cv_running = False
                    for vi in range(1,6):
                        if validation_ndcg[-vi] > validation_ndcg[-6]:
                            cv_running = True
                            break
            else:
                validation_ndcg.append(0)
        
        
            with open(results_file, 'a+') as f:
                json.dump({'jobid':jobid, 'model':loss_fn_str, 'dataset':dataset_name, 
                           'train_queries':dataset.trdlr.shape[0]-1, 'train_docs':dataset.trlv.shape[0],
                           'layers':str(layers), 'epoch':epoch+1, 
                           'start_time':start_time, 'current_time':int(time.time()),
                           'seed':rseed, 'learning_rate':learning_rate,
#                            'train':list(results['train']), 'valid':list(results['valid']),
                           'test':list(results['test']), 
                           'opt': optimizer_str, 
                           'first_layers_train_epochs':first_layers_train_epochs,
                           'ndcg@10':results['test'][-1],
                           'CDR_ndcg':cdr_eval,
                           'EB':compute_EB(Grads),
                           'GD':compute_GD(Grads),
                           'EB_last':compute_EB(Grads_last),
                           'GD_last':compute_GD(Grads_last),
                           'CDR':cosine(alphas[0],alphas[1]),
                           'CV_ndcg':validation_ndcg[-1],

                           #'weights':net.weight().data.cpu().numpy().tolist(),
                            }, f)
                f.write('\n')
