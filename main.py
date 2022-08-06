'''
Created on 25 May 2021

@author: aliv
'''
from data_util import *
from models import *
from losses import *

from absl import app
from absl import flags
import os
import sys
from builtins import isinstance

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'yahoo', 'dataset pickle address')
    flags.DEFINE_string('layers', '[1000,1000]', 'layers of the DNN')
    flags.DEFINE_string('epochs', '50', 'num of epochs')
    flags.DEFINE_string('results', '/ivi/ilps/personal/avardas/_data/overfit/', 'results address')
    flags.DEFINE_string('loss', 'lambdaRank', 'loss function: "lambdaRank", "ordinal", "rmse", "listNet"')
    flags.DEFINE_string('opt', 'SGD', 'Adam or SGD')
    flags.DEFINE_string('learning_rate', '0.01', 'learning rate')
    flags.DEFINE_string('jobid', '777', 'job ID')
    flags.DEFINE_string('l2', '0', '')
    flags.DEFINE_integer('toy_size', 10, 'number of training samples for a toy test.')
    flags.DEFINE_string('subsample_rseed', '7', 'random seed for subsampling of training samples for a toy test.')
    flags.DEFINE_string('first_layers_train_epochs', '0', '')
    flags.DEFINE_string('cv_folds', '5', '')
    
  
  
def main(args):
    datapath = {'yahoo':'/ivi/ilps/personal/avardas/_data/ltrc_yahoo/set1.binarized_purged_querynorm_filtered.npz',
    #               'yahoo_small_8':'/ivi/ilps/personal/avardas/_data/ltrc_yahoo/set1.binarized_purged_querynorm_small_8.npz',
    #               'yahoo_small_2':'/ivi/ilps/personal/avardas/_data/ltrc_yahoo/set1.binarized_purged_querynorm_small_2.npz',
    #               'yahoo_small_4':'/ivi/ilps/personal/avardas/_data/ltrc_yahoo/set1.binarized_purged_querynorm_small_4.npz',
              'mslr':'/ivi/ilps/personal/avardas/_data/MSLR-WEB30k/Fold1/binarized_purged_querynorm_filtered.npz',
              's_mslr':'/ivi/ilps/personal/avardas/_data/MSLR-WEB30k/Fold1/binarized_purged_querynorm_small.npz'}[FLAGS.dataset]

    layers = eval(FLAGS.layers)
    epochs = eval(FLAGS.epochs)
    loss = FLAGS.loss
    learning_rate = eval(FLAGS.learning_rate)
    subsample_rseed = eval(FLAGS.subsample_rseed)

    if not isinstance(learning_rate, list):
        learning_rate = [learning_rate]
    if not isinstance(subsample_rseed, list):
        subsample_rseed = [subsample_rseed]


    for rseed in subsample_rseed:
        dataset = read_pkl(datapath, FLAGS.toy_size, rseed)
        for lr in learning_rate:
            train_model(jobid = FLAGS.jobid, 
                        dataset = dataset, 
                        dataset_name = FLAGS.dataset, 
                        layers = layers,
                        epochs = epochs, learning_rate = lr, l2 = eval(FLAGS.l2),
                        rseed = rseed+777,
                        optimizer_str = FLAGS.opt, loss_fn_str = FLAGS.loss,
                        first_layers_train_epochs = eval(FLAGS.first_layers_train_epochs),
                        cv_folds = eval(FLAGS.cv_folds),
                        results_file = FLAGS.results)


if __name__ == '__main__':
    app.run(main)
