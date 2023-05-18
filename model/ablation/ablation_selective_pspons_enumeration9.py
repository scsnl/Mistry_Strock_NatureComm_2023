import torch
import pytorch_lightning as pl
from common.model.mycornet import ExtendedCORnet, ZeroMask, ShuffleMask
from common.dataset.enumeration import CountingDotsDataSet
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import pandas as pd
import os, psutil
process = psutil.Process(os.getpid())
from humanfriendly import format_size
import argparse
from functools import reduce
import sys
from pprint import pprint
from scipy.io import loadmat


def main(args):

    seed = 0
    pl.seed_everything(seed)

    # -------------------------------
    # Parameters
    # -------------------------------

    n_max = 9
    n_sample_train = 10 # number of samples used in training per class/condition
    n_sample_test = 2 # number of samples used in test per class/condition
    sample = np.arange(n_sample_train, n_sample_train+n_sample_test) # id of sample used

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where accuracies are saved
    accuracy_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/ablation'
    # path where log of training are saved
    log_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/log/ablation'
    # path containing the dataset
    dataset_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/stimuli'
    # path containing the model
    model_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/model'
    # path containing ids of sensitive neurons to numerosity
    sensitivity_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/sensitivity'

    # -------------------------------
    # Test dataset
    # -------------------------------

    # all condition used
    pattern_condition = "\\w*"
    # all param used
    pattern_param = "(\\w|-)*"
    # only the selected sample are used
    pattern_sample = f'\\d*_({reduce(lambda a,b: f"{a}|{b}", sample)})'
    # simple global pattern containing files
    path = f'{dataset_path}/*/*/*.png'
    # regex matching only the files that should be used for training
    include = f'{dataset_path}/{pattern_condition}/{pattern_param}/{pattern_sample}.png'

    dataset = CountingDotsDataSet(path = path, include = include)
    test_loader = DataLoader(dataset, batch_size = 50, shuffle = False, num_workers = 4, pin_memory = True)

    # -------------------------------
    # Saving model
    # -------------------------------

    epochs = args.epochs
    accuracy = np.zeros((len(epochs),2))
    trainer = pl.Trainer(default_root_dir=log_path, deterministic=True, devices="auto", accelerator="auto")
    model  = ExtendedCORnet(out_features = n_max)

    # -------------------------------
    # Testing model
    # -------------------------------

    _spontaneous = np.load(f'{sensitivity_path}/epoch-1/IT_sensitive.npz')["sensitive"][-1]
    spontaneous = np.zeros(_spontaneous.size, dtype = _spontaneous.dtype)
    _idx = loadmat(f'{os.environ.get("OAK")}/projects/percym/2021_DNN_1_9/results/dnn/dnn_astrock_31Jan2022/SPON_only_Sep2022.mat')['SPONIPS']-1
    spontaneous[_idx] = 1
    spontaneous = spontaneous.reshape(_spontaneous.shape, order = 'F')
    _preferred = loadmat(f'{os.environ.get("OAK")}/projects/percym/2021_DNN_1_9/results/dnn/dnn_astrock_31Jan2022/outsummary/dnn_astrock_31Jan22_IT_-1.mat')['D']['maxN2'][0,0][:,0]

    for i, epoch in enumerate(epochs):
        checkpoint = torch.load(f'{model_path}/epoch{epoch:02}.ckpt')
        model.load_state_dict(checkpoint['state_dict'])
        persistent = np.zeros(_spontaneous.size, dtype = _spontaneous.dtype)
        D = loadmat(f'{os.environ.get("OAK")}/projects/percym/2021_DNN_1_9/results/dnn/dnn_astrock_31Jan2022/outsummary/dnn_astrock_31Jan22_IT_{epoch:02d}.mat')['D']
        idx = (D['numOnly2'][0,0][:,0])>0
        persistent[idx] = 1
        persistent = persistent.reshape(_spontaneous.shape, order = 'F')
        preferred = D['maxN2'][0,0][:,0]
        persistent = persistent&spontaneous&(preferred == _preferred).reshape(_spontaneous.shape, order = 'F')
        IT0 = model.model.IT
        model.model.IT = torch.nn.Sequential(IT0, ZeroMask(persistent))
        metrics, = trainer.test(model, test_loader)
        accuracy[i,0] = metrics['test_acc']
        model.model.IT = torch.nn.Sequential(IT0, ZeroMask(np.logical_not(persistent)))
        metrics, = trainer.test(model, test_loader)
        accuracy[i,1] = metrics['test_acc']

    epoch_accuracy = np.zeros(len(epochs), dtype = np.dtype([('epoch', np.int64), ('accuracy', np.float64, 2)]))
    epoch_accuracy['epoch'] = epochs
    epoch_accuracy['accuracy'] = accuracy
    os.makedirs(f'{accuracy_path}', exist_ok=True)
    np.save(f'{accuracy_path}/persistentv2_epochs_{epochs}_accuracy.npy', epoch_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test models at different epochs')
    parser.add_argument('--epochs', metavar = 'E', type = int, nargs = "+", help = 'list of epochs to test')
    args = parser.parse_args()
    main(args)