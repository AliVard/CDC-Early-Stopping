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
    


def get_torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_mnist():
        # trainset = torchvision.datasets.EMNIST(root='/ivi/ilps/personal/avardas/_data/torchvision', split='digits', train=True
    trainset = torchvision.datasets.MNIST(root='/ivi/ilps/personal/avardas/_data/torchvision', train=True
        ,transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1732,), (0.3317,))
        ])
    )
    # noise = torch.randint(1,10, trainset.targets.shape) * torch.bernoulli(0.7*torch.ones(trainset.targets.shape))
    # trainset.targets = (trainset.targets + noise) % 10

#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False)
    # testset = torchvision.datasets.EMNIST(root='/ivi/ilps/personal/avardas/_data/torchvision', split='digits', train=False
    testset = torchvision.datasets.MNIST(root='/ivi/ilps/personal/avardas/_data/torchvision', train=False
        ,transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1732,), (0.3317,))
        ])
    )
    
    return trainset, testset

def load_cifar10():
    trainset = torchvision.datasets.CIFAR10(root='/ivi/ilps/personal/avardas/_data/torchvision/CIFAR', train=True,
                                            transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
        ]))

    testset = torchvision.datasets.CIFAR10(root='/ivi/ilps/personal/avardas/_data/torchvision/CIFAR', train=False,
                                           transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
        ]))
    return trainset, testset
    
def load_data(dataset_name, sub_sample_ratio = 20, noise_level = 0.8):

    if dataset_name == 'mnist':
        trainset, testset = load_mnist()
    elif dataset_name == 'cifar10':
        trainset, testset = load_cifar10()
    else:
        print(dataset_name + ' data loader not implemented')
        return

    unique_labels = np.unique(trainset.targets)

    selected_ids = []
    for l in unique_labels:
        l_ind = np.where(trainset.targets == l)[0]
        selected_ids.append(np.random.choice(l_ind, int(l_ind.shape[0]/sub_sample_ratio)))
    selected_ids = np.sort(np.concatenate(selected_ids, 0))
    trainset.data = trainset.data[selected_ids, :, :]
    trainset.targets = torch.tensor(trainset.targets)[selected_ids].type(torch.LongTensor)
    testset.targets = torch.tensor(testset.targets).type(torch.LongTensor)


    print('subsampled train:', len(trainset))


    if noise_level > 0:
        noise = torch.randint(1,10, trainset.targets.shape) * torch.bernoulli(noise_level*torch.ones(trainset.targets.shape))
        trainset.targets = (trainset.targets + noise) % 10
    
    return trainset, testset



def smooth(vec):
    vec = np.insert(vec, 0, vec[0])
    vec = np.append(vec, vec[-1])
    s = vec[1:-1]/2 + vec[:-2]/4 + vec[2:]/4
    return s


def smooth5(vec):
    m = vec.mean()
    d = np.diff(vec, prepend=2*vec[0]-vec[1])
    for i in range(5):
        d = smooth(d)
    nvec = np.cumsum(d)
    nm = nvec.mean()
    nvec += m - nm
    return nvec


def curvature_curve(cos, epochs, normalizer):
    cos = smooth5(cos)
    cos_dif = smooth5(np.diff(cos, append=0))
    e_dif = np.diff(epochs/normalizer, append=0)
    cos_dif_dif = smooth5(np.diff(cos_dif, append=0))
    e_dif_dif = np.diff(e_dif, append=0)
#     curve = np.abs((e_dif_dif*cos_dif - cos_dif_dif*e_dif)/(e_dif**2 + cos_dif**2)**(1.5)) * np.sign(cos_dif_dif)
#     curve = np.abs((e_dif_dif*cos_dif - cos_dif_dif*e_dif)/(e_dif**2 + cos_dif**2)**(1.5))
    curve = (cos_dif_dif / normalizer) / (1./normalizer**2 + cos_dif**2)**1.5
    curve[:2] = curve[2]
    curve[-20:] = curve[-20]
#     print(curve)
#     return curve
    return smooth5(curve)

def curvature(cos, epochs, normalizer):
    return curvature_curve(cos, epochs, normalizer).argmax()





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
    

    

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=None):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_layers[0], bias=False)]
        for i in range(1, len(hidden_layers)):
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_layers[-1], output_dim, bias=False))
        
        self.layers = nn.Sequential(*layers)
        
        
        if torch.cuda.is_available():
            self.cuda(get_torch_device())
            
        
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
        
        y = y.type(torch.LongTensor)
        
        x = x.to(get_torch_device())
        y = y.to(get_torch_device())
            
        tx = torch.flatten(x, start_dim=1)
        out = self(tx)#[:,0]
        
#         print(f'tx: {tx.shape}')

#         print(f'out: {out.shape}')
#         print(f'y: {y.shape}')
        
#         print(out)
#         print('y:', y)

        #out = out.type(torch.LongTensor)
        loss = loss_fn(out, y)
        loss.backward()
        self.opt.step()
            
        grads1_s = []
        
        for param in self.w:
            grads1_s.append(param.grad.view(-1))
        grads1_s = torch.cat(grads1_s)
        return grads1_s.data.cpu().numpy()[None,:]
      
    
    def params(self, only_last=False):
        self.only_last = only_last
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


def eval_model(epoch, epochs):
    if epoch < 10 or epoch % 5 == 4 or epoch == epochs-1:
        return True
    return False


def evaluate(net, testset, batch_size):
    correct = 0
    total = 0
    net.eval()

    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

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
        
    
def train(net, trainset, testset, epochs, layers=None, lr=None, l2=0, only_last=True):
    
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    
    unique_labels = len(np.unique(trainset.targets))
    
    if net is None:
        initargs = (trainset.data[0].flatten().shape[0], layers, unique_labels)
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
            net.evals.append(evaluate(net, testset, 512))
            net.epochs.append(epoch)
    return net
            
        
    
def train_copy(net, trainset, override = True):
    
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    
    netc = DNN(*net.initargs)
    netc.opt = torch.optim.SGD(netc.params(only_last=override), lr=net.lr, weight_decay=net.l2)
    if override:
        netc.override_first_layers(net)
    netc.weights_vectors = [[] for _ in range(len(net.w))]
    
    for epoch in range(len(net.weights_vectors[0])):
        
        netc.train()
        for (x, y) in train_dl:
            grads_s = netc.train_batch(loss_fn, x, y)
            
        for i in range(len(netc.w)):
            netc.weights_vectors[i].append(netc.w[i].data.cpu().numpy().copy())
            
    return netc
            
def get_cos(weights_vectors1, weights_vectors2):

    cos = [[] for _ in range(weights_vectors1[0].shape[0])]
    for l in range(len(cos)):
        for j in range(len(weights_vectors1)):
            cos[l].append(cosine(weights_vectors1[j][l,:], weights_vectors2[j][l,:]))
    return cos

def get_cos_initial(weights_vectors1):

    cos = [[] for _ in range(weights_vectors1[0].shape[0])]
    for l in range(len(cos)):
        for j in range(len(weights_vectors1)):
            cos[l].append(cosine(weights_vectors1[0][l,:], weights_vectors1[j][l,:]))
    return cos

def get_X(trainset):
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    for (x, y) in train_dl:
        return torch.flatten(x, start_dim=1).data.cpu().numpy(), y.data.cpu().numpy()
    
def get_activations(X, net):
    activations = [[] for _ in range(len(net.weights_vectors) - 1)]
    
    for i in range(len(net.weights_vectors[0])):
        activations[0].append(np.matmul(X, net.weights_vectors[0][i].T))
    for layer in range(1, len(activations)):
        for i in range(len(net.weights_vectors[layer])):
            activations[layer].append(np.matmul(activations[layer-1][i], net.weights_vectors[layer][i].T))
        
    return activations

def transform_weights(weights, A1, A2):
    tw = []
    for i in range(len(A1)):
        a = np.linalg.pinv(A1[i])
        tw.append(np.matmul(
                            np.matmul(a, A2[i]), 
                            weights[i].T).T
                 )
    return tw

def transform_weights_2(X, net, netc):

    weights = netc.weights_vectors[1]
    tw = []
    
#     print(len(net.weights_vectors[0]))
    for i in range(len(net.weights_vectors[0])):
        A1 = np.matmul(X, net.weights_vectors[0][i].T)
        A2 = np.matmul(X, netc.weights_vectors[0][i].T)
        
        a = np.linalg.pinv(A1)
        tw.append(np.matmul(
                            np.matmul(a, A2), 
                            weights[i].T).T
                 )
    return tw

    

def main():
    trainset, testset = load_data(flag_dataset, sub_sample_ratio=1, noise_level=.5)



    # flag_lr = eval(sys.argv[1])
    # flag_layers = eval(sys.argv[2])
    # flag_lastlayer = eval(sys.argv[3])
    # flag_savepkl = sys.argv[4]


    net = train(None, trainset=trainset, testset=testset, epochs=flag_epochs, layers=flag_layers, lr=flag_lr, only_last=flag_lastlayer)
    netc = train_copy(net, trainset=trainset, override=flag_lastlayer)

    X, Y = get_X(trainset)

    tmp = {'net': net, 'netc': netc, 'X': X, 'Y': Y}
    if flag_lastlayer:
        ys, ycs = [], []
        A = X @ list(net.parameters())[0].data.cpu().numpy().T
        A[A<0] = 0
        Ac = X @ list(netc.parameters())[0].data.cpu().numpy().T
        Ac[Ac<0] = 0
        for i in range(len(net.weights_vectors[0])):
            ycs.append(Ac @ netc.weights_vectors[0][i].T)
            ys.append(A @ net.weights_vectors[0][i].T)
        tmp['y'] = ys
        tmp['yc'] = ycs
    else:
        ys, ycs, Byc = [], [], []
        for i in range(len(net.weights_vectors[0])):
            A = X
            Ac = X
            for layer in range(len(net.weights_vectors) - 1):
                A = A @ net.weights_vectors[layer][i].T
                A[A<0] = 0
                Ac = Ac @ netc.weights_vectors[layer][i].T
                Ac[Ac<0] = 0
            B = np.linalg.pinv(A)
            yc = Ac @ netc.weights_vectors[-1][i].T
            Byc.append((B @ yc).T)
            ycs.append(yc)
            ys.append(A @ net.weights_vectors[-1][i].T)
        tmp['Byc'] = Byc
        tmp['y'] = ys
        tmp['yc'] = ycs
        
        

    with open(flag_savepkl, 'wb') as f:
        pickle.dump(tmp, f)
        
        
if __name__ == '__main__':
    main()
