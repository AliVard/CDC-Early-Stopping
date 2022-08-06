

from sklearn.model_selection import KFold
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import pickle
import os
from scipy.spatial.distance import cosine

import sys

from GeneralML import *

if __name__ == '__main__':
    flag_lr = eval(sys.argv[1])
    flag_layers = eval(sys.argv[2])
    flag_lastlayer = eval(sys.argv[3])
    flag_savepkl = sys.argv[4]

    flag_epochs = 100
    if len(sys.argv) > 5:
        flag_epochs = eval(sys.argv[5])
        
    flag_dataset = 'mnist'
    if len(sys.argv) > 6:
        flag_dataset = sys.argv[6]
        
    flag_subsample = 1
    if len(sys.argv) > 7:
        flag_subsample = eval(sys.argv[7])
        
    flag_noise = 0.5
    if len(sys.argv) > 8:
        flag_noise = eval(sys.argv[8])
    



def evaluate_loader(net, testloader):
    correct = 0
    total = 0
    net.eval()

    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(get_torch_device())
            y = y.to(get_torch_device())
            tx = torch.flatten(x, start_dim=1)
            output = net(tx)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total +=1
    return correct/total

def train_loader(net, input_size, output_size, train_dl, testloader, epochs, layers=None, lr=None, l2=0, only_last=True):
    
    loss_fn = nn.CrossEntropyLoss()
    
    if net is None:
        initargs = (input_size, layers, output_size)
        net = DNN(*initargs)
        net.initargs = initargs
        net.lr = lr
        net.l2 = l2
        net.opt = torch.optim.SGD(net.params(only_last=only_last), lr=lr, weight_decay=l2)
        net.evals = []
        net.epochs = []
        net.eb = []
        net.gd = []
        net.weights_vectors = [[] for _ in range(len(net.w))]
        epochs_from = 0
    else:
        epochs_from = net.epochs[-1] + 1
    
    
    for epoch in range(epochs_from, epochs_from + epochs):
        
        batch = 0
        ndcgs = []
        net.train()
        Grads = []
        for (x, y) in train_dl:
            grads_s = net.train_batch(loss_fn, x, y)
            Grads.append(grads_s)
            
        
        Grads = np.concatenate(Grads, axis=0)
        net.eb.append(compute_EB(Grads))
        net.gd.append(compute_GD(Grads))
        for i in range(len(net.w)):
            net.weights_vectors[i].append(net.w[i].data.cpu().numpy().copy())
        if eval_model(epoch, epochs_from + epochs):
            net.evals.append(evaluate_loader(net, testloader))
            net.epochs.append(epoch)
    return net

def main():
    kfold = KFold(n_splits=5, shuffle=True)
    
    trainset, testset = load_data(flag_dataset, sub_sample_ratio=flag_subsample, noise_level=flag_noise)

    output_size = len(np.unique(trainset.targets))
    input_size = trainset.data[0].flatten().shape[0]
    
    evals = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainset)):

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          trainset, 
                          batch_size=256, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          trainset,
                          batch_size=256, sampler=test_subsampler)

        net = train_loader(None, input_size, output_size, trainloader, testloader, epochs=flag_epochs, layers=flag_layers, lr=flag_lr, only_last=flag_lastlayer)

        evals.append(net.evals)

    
    with open(flag_savepkl+'.cv', 'wb') as f:
        pickle.dump({'evals':evals, 'epochs':net.epochs}, f)
    

if __name__ == '__main__':
    main()
