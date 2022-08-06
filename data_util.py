import time
import lightgbm as lgb
import metrics
import os
import numpy as np
import json

import pickle

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def _subsample_by_ids(dlr, fm, lv, qids):
    feature_matrix = []
    label_vector = []
    doclist_ranges = [0]
    for qid in qids:
        s_i = dlr[qid]
        e_i = dlr[qid+1]
        feature_matrix.append(fm[s_i:e_i, :])
        label_vector.append(lv[s_i:e_i])
        doclist_ranges.append(e_i - s_i)
    
    doclist_ranges = np.cumsum(np.array(doclist_ranges), axis=0)
    feature_matrix = np.concatenate(feature_matrix, axis=0)
    label_vector = np.concatenate(label_vector, axis=0)
    return doclist_ranges, feature_matrix, label_vector
    
def _random_subsample(dlr, fm, lv, subsample_size, seed):
    np.random.seed(seed)
    qids = np.random.randint(0, dlr.shape[0]-1, subsample_size)
    
    return _subsample_by_ids(dlr, fm, lv, qids)
    

def train_random_subsample(dataset, subsample_size, seed):
    np.random.seed(seed)
    qids = np.random.randint(0, dataset.trdlr.shape[0]-1, subsample_size)
    
    sub_ds = type('', (), {})()
    sub_ds.trdlr, sub_ds.trfm, sub_ds.trlv = _subsample_by_ids(dataset.trdlr, dataset.trfm, dataset.trlv, qids)
    return sub_ds
    
    
def subsample_dataset(dataset, subsample_size, seed):
    sdataset = type('', (), {})()
    sdataset.trdlr, sdataset.trfm, sdataset.trlv = _random_subsample(dataset.trdlr, 
                                                                     dataset.trfm, 
                                                                     dataset.trlv, 
                                                                     subsample_size, seed)
    vasize = int(subsample_size * (dataset.vadlr.shape[0]-1) / (dataset.trdlr.shape[0]-1))
    sdataset.vadlr, sdataset.vafm, sdataset.valv = _random_subsample(dataset.vadlr, 
                                                                     dataset.vafm, 
                                                                     dataset.valv, 
                                                                     vasize, seed)
    tesize = int(subsample_size * (dataset.tedlr.shape[0]-1) / (dataset.trdlr.shape[0]-1))
    sdataset.tedlr, sdataset.tefm, sdataset.telv = _random_subsample(dataset.tedlr, 
                                                                     dataset.tefm, 
                                                                     dataset.telv, 
                                                                     tesize, seed)
    return sdataset
    

def kfold_train(dataset, k=5):
    folds = [type('', (), {})() for _ in range(k)]
    
    
    docs_len = dataset.trdlr.shape[0] - 1
    step_len = int(docs_len/k)
    
    for i in range(len(folds)):
        trqids = []
        trqids += list(range(i*step_len))
        trqids += list(range((i+1)*step_len, docs_len))
        
        folds[i].trdlr, folds[i].trfm, folds[i].trlv = _subsample_by_ids(dataset.trdlr, dataset.trfm, dataset.trlv, trqids)
        folds[i].vadlr, folds[i].vafm, folds[i].valv = _subsample_by_ids(dataset.trdlr, dataset.trfm, dataset.trlv, range(i*step_len, (i+1)*step_len))

        
    return folds
           
def remove_duplicates(dataset):
    dlr, fm, lv = [0], [], []
    for qid in range(dataset.trdlr.shape[0] - 1):
        s_i, e_i = dataset.trdlr[qid:qid+2]
        x = dataset.trfm[s_i:e_i,:]
        y = dataset.trlv[s_i:e_i]
        diff = x[:,None,:] - x[None,:,:]
        diff = diff.sum(2)
        diff += np.tril(np.ones(diff.shape))
        uniques = np.ones(diff.shape[0], dtype=np.bool8)
        uniques[np.where(diff==0)[0]] = False
#         print(uniques)
        fm.append(x[uniques,:])
        lv.append(y[uniques])
        dlr.append(sum(uniques))
    return np.concatenate(fm,0), np.concatenate(lv,0), np.cumsum(np.array(dlr))

def read_pkl(pkl_path, toy_size=-1, subsample_rseed=0):
    loaded_data = np.load(pkl_path, allow_pickle=True)
#     feature_map = loaded_data['feature_map'].item()
    train_feature_matrix = loaded_data['train_feature_matrix']
    train_doclist_ranges = loaded_data['train_doclist_ranges']
    train_label_vector   = loaded_data['train_label_vector']
    valid_feature_matrix = loaded_data['valid_feature_matrix']
    valid_doclist_ranges = loaded_data['valid_doclist_ranges']
    valid_label_vector   = loaded_data['valid_label_vector']
    test_feature_matrix  = loaded_data['test_feature_matrix']
    test_doclist_ranges  = loaded_data['test_doclist_ranges']
    test_label_vector    = loaded_data['test_label_vector']
    dataset = type('', (), {})()
#     dataset.fmap = feature_map
    dataset.trfm = train_feature_matrix
    dataset.tefm = test_feature_matrix
    dataset.vafm = valid_feature_matrix
    dataset.trdlr = train_doclist_ranges
    dataset.tedlr = test_doclist_ranges
    dataset.vadlr = valid_doclist_ranges
    dataset.trlv = train_label_vector
    dataset.telv = test_label_vector
    dataset.valv = valid_label_vector
    
    if toy_size > 0:
      sub_ds = train_random_subsample(dataset, toy_size, subsample_rseed)
      dataset.trdlr = sub_ds.trdlr
      dataset.trfm = sub_ds.trfm
      dataset.trlv = sub_ds.trlv
    dataset.trfm, dataset.trlv, dataset.trdlr = remove_duplicates(dataset)
    
    print('num features : {}'.format(dataset.trfm.shape[1]))
    print('num docs (train, valid, test) : ({},{},{})'.format(dataset.trfm.shape[0], dataset.vafm.shape[0], dataset.tefm.shape[0]))
    print('num queries (train, valid, test) : ({},{},{})'.format(dataset.trdlr.shape[0], dataset.vadlr.shape[0], dataset.tedlr.shape[0]))

    return dataset
    

def lambdarank(dataset, model_path=None, learning_rate=0.05, n_estimators=300, eval_at=[10], early_stopping_rounds=10000):
    start = time.time()
    if model_path is not None and os.path.exists(model_path):
        booster = lgb.Booster(model_file=model_path)
        print('loading lgb took {} secs.'.format(time.time() - start))
        return booster.predict

    gbm = lgb.LGBMRanker(learning_rate=learning_rate, n_estimators=n_estimators)

    gbm.fit(dataset.trfm, dataset.trlv, 
          group=np.diff(dataset.trdlr), 
          eval_set=[(dataset.vafm, dataset.valv)],
          eval_group=[np.diff(dataset.vadlr)], 
          eval_at=eval_at, 
          early_stopping_rounds=early_stopping_rounds, 
          verbose=False)

    if model_path is not None:
        gbm.booster_.save_model(model_path)

    print('training lgb took {} secs.'.format(time.time() - start))
    return gbm.booster_.predict


def gbmregressor(dataset, model_path=None, learning_rate=0.05, n_estimators=300, early_stopping_rounds=10000):
    start = time.time()
    if model_path is not None and os.path.exists(model_path):
        booster = lgb.Booster(model_file=model_path)
        print('loading lgb took {} secs.'.format(time.time() - start))
        return booster.predict

    gbm = lgb.LGBMRegressor(learning_rate=learning_rate, n_estimators=n_estimators)

    gbm.fit(dataset.trfm, dataset.trlv,
          eval_set=[(dataset.vafm, dataset.valv)],
          early_stopping_rounds=early_stopping_rounds, 
          verbose=False)

    if model_path is not None:
        gbm.booster_.save_model(model_path)

    print('training lgb took {} secs.'.format(time.time() - start))
    return gbm.booster_.predict


def clf_pred(booster):
    return lambda x: booster.predict(x).argmax(axis=1)
'''
def gbmclassifier(dataset, model_path=None, learning_rate=0.05, n_estimators=300, early_stopping_rounds=10000):
    start = time.time()
    if model_path is not None and os.path.exists(model_path):
        booster = lgb.Booster(model_file=model_path)
        print('loading lgb took {} secs.'.format(time.time() - start))
        return clf_pred(booster)

    gbm = lgb.LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators)

    gbm.fit(dataset.trfm, dataset.trlv,
          eval_set=[(dataset.vafm, dataset.valv)],
          early_stopping_rounds=early_stopping_rounds, 
          verbose=False)

    if model_path is not None:
        gbm.booster_.save_model(model_path)

    print('training lgb took {} secs.'.format(time.time() - start))
    return clf_pred(gbm.booster_)



def evaluate(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.tefm)
    if saveto is not None:
        results = {}
        results['dlr'] = dataset.tedlr
        results['lv'] = dataset.telv
        results['pred'] = y_pred
        with open(saveto, 'wb') as f:
            pickle.dump(results, 
                        f)
    metric = metrics.LTRMetrics(dataset.telv,np.diff(dataset.tedlr),y_pred)

    return metric.NDCG(1), metric.NDCG(3), metric.NDCG(5), metric.NDCG(10)

def evaluate_valid(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.vafm)
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
    y_pred = predictor(dataset.trfm)
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



class LTRData(Dataset):
    def __init__(self, allfm, alllv):
        self.features = torch.FloatTensor(allfm)
        self.labels = torch.FloatTensor(alllv)
    
    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]
    
class QueryGroupedLTRData(Dataset):
    def __init__(self, fm, dlr, lv):
        self.feature = fm
        self.dlr = dlr
        self.labels = lv
    
    def __len__(self):
        return self.dlr.shape[0] - 1

    def __getitem__(self, qid):
        s_i = self.dlr[qid]
        e_i = self.dlr[qid+1]
        feature = torch.FloatTensor(self.feature[s_i:e_i,:])
        labels = torch.FloatTensor(self.labels[s_i:e_i])
        return feature, labels

def listwise_loss(scores, labels):
    sigma = 1.
    rank_df = pd.DataFrame({"Y": labels, "doc": np.arange(labels.shape[0])})
    rank_df = rank_df.sort_values("Y").reset_index(drop=True)
    rank_order = rank_df.sort_values("doc").index.values + 1


    with torch.no_grad():
        pos_pairs_score_diff = 1.0 + torch.exp(sigma * (scores - scores.t()))

        Y_tensor = torch.FloatTensor(labels).view(-1, 1)
        rel_diff = Y_tensor - Y_tensor.t()
        pos_pairs = (rel_diff > 0).type(torch.float)
        neg_pairs = (rel_diff < 0).type(torch.float)

        Sij = pos_pairs - neg_pairs
        
        gain_diff = torch.pow(2.0, Y_tensor) - torch.pow(2.0, Y_tensor.t())
        rank_order_tensor = torch.FloatTensor(rank_order).view(-1, 1)
        decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)
        delta_dcg = torch.abs(gain_diff * decay_diff)       
        
        lambda_update = sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_dcg


        lambda_update = torch.sum(lambda_update, 1, keepdim=True)
        
    return lambda_update
        
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=False):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            if dropout:
                layers.append(nn.Dropout(p=0.5, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class DNN_mean(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=False):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            if dropout:
                layers.append(nn.Dropout(p=0.5, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.mean(self.layers(x), dim=1)

def seed(random_seed):
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed) 
    
def train_dnn_list(dataset, layers, epochs, results_file, print_evals=True):
#    seed(7)
#     train_dl = DataLoader(LTRData(dataset.trfm, dataset.trlv), batch_size=512, shuffle=True)
    train_dl = DataLoader(QueryGroupedLTRData(dataset.trfm, dataset.trdlr, dataset.trlv), batch_size=1, shuffle=True)
    net = DNN(dataset.trfm.shape[1], layers)
#    print(layers)
#    print(net.layers[0].weight)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        batch = 0
        net.train()
        for (x, y) in train_dl:
            optimizer.zero_grad()
#             out = net(x)[:,0]
#             loss = nn.functional.mse_loss(out, y)
#             loss.backward()
            scores = net(x[0])
#             print(scores.shape)
#             print(y[0].shape)
            with torch.no_grad():
                lambda_i = listwise_loss(scores, y[0])
            torch.autograd.backward(scores, lambda_i)

            optimizer.step()
#            if batch < 20:
#                print(net.layers[0].weight)
#                print('-'*20)
#                batch += 1 

        if print_evals:
            results = {}
            results['train'] = evaluate_train(dataset, predict_dnn(net.eval()))
            results['test'] = evaluate(dataset, predict_dnn(net.eval()))
            with open(results_file, 'a+') as f:
                json.dump({'model':'DNN_list', 'train_size':dataset.trdlr.shape[0]-1, 'layers':layers, 'epoch':epoch+1, 'results':results}, f)
                f.write('\n')
    return net


def train_dnn(dataset, layers, epochs, results_file, dropout=False, print_evals=True):
    train_dl = DataLoader(LTRData(dataset.trfm, dataset.trlv), batch_size=512, shuffle=True)
    # net = DNN(dataset.trfm.shape[1], layers, dropout)
    net = DNN_mean(dataset.trfm.shape[1], layers, dropout)
    model_name = 'pointwise_' + str(layers).replace(',','-').replace(' ','').replace('[','').replace(']','') + '_'
    if dropout:
        model_name += 'dropout_'
    model_name += str(epochs)
    if os.path.exists(model_name + '.mdl'):
        net.load_state_dict(torch.load(model_name + '.mdl'))
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            batch = 0
            net.train()
            for (x, y) in train_dl:
                optimizer.zero_grad()
                out = net(x)#[:,0]
                # print(out.shape)
                # print(y.shape)
                loss = nn.functional.mse_loss(out, y)
                loss.backward()

                optimizer.step()
            if print_evals:
                results = {}
                results['train'] = evaluate_train(dataset, predict_dnn(net.eval()))
                results['test'] = evaluate(dataset, predict_dnn(net.eval()))
                print({'model':'DNN', 'train_size':dataset.trdlr.shape[0]-1, 'layers':layers, 'epoch':epoch+1, 'results':results})
                with open(results_file, 'a+') as f:
                    json.dump({'model':model_name, 'train_size':dataset.trdlr.shape[0]-1, 'layers':layers, 'epoch':epoch+1, 'results':results}, f)
                    f.write('\n')
        torch.save(net.state_dict(), model_name + '.mdl')
    evaluate_train(dataset, predict_dnn(net.eval()), model_name + '_train.out')
    evaluate(dataset, predict_dnn(net.eval()), model_name + '_test.out')
    return net



def predict_dnn(net):
    def predict(x):
        x = torch.FloatTensor(x)
        with torch.no_grad():
            output = net(x)
            return output.squeeze().numpy()
    return predict


def multiple_models(dataset, model, subsample_size, random_seeds, trees_list, results_file):
    model_fn = {'lambdarank':lambdarank, 'regressor':gbmregressor, 'classifier':gbmclassifier}[model]
    for n_estimators in trees_list:
        for seed in random_seeds:
            sdataset = subsample_dataset(dataset, subsample_size, seed)
            booster = model_fn(sdataset, n_estimators=n_estimators)
#             booster = lambdarank(sdataset, n_estimators=n_estimators, model_path=f'models/seed{seed}_trees{n_estimators}.gbm')
            results = {}
            results['train'] = evaluate_train(sdataset, booster)
            results['valid'] = evaluate_valid(sdataset, booster)
            results['test'] = evaluate(sdataset, booster)
            with open(results_file, 'a+') as f:
                json.dump({'model':model, 'train_size':subsample_size, 'trees':n_estimators, 'seed':seed, 'results':results}, f)
                f.write('\n')


'''
