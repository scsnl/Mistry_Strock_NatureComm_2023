import torch
import pytorch_lightning as pl
from nn_modeling.model.torch import ExtendedCORnet
from torchvision.datasets import ImageNet, ImageFolder
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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def main(args):

    seed = 0
    pl.seed_everything(seed)

    # -------------------------------
    # Parameters
    # -------------------------------

    n_max = 9

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where accuracies are saved
    accuracy_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/accuracy/cornet_adam'
    # path where log of training are saved
    log_path = f'{os.environ.get("TMP_PATH")}/enumeration_{n_max}/log/test'
    # path containing the dataset
    dataset_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/stimuli'
    # path containing the model
    model_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/model/cornet_adam'

    # -------------------------------
    # Test dataset
    # -------------------------------

    transform = Compose([
                    Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])

    dataset = ImageNet(root = f'{os.environ.get("NN_COMMON")}/data/dataset/imagenet', split = 'val', transform = transform)
    #dataset = ImageFolder(root = f'{os.environ.get("NN_COMMON")}/data/dataset/imagenet/val', transform = transform)
    test_loader = DataLoader(dataset, batch_size = 50, shuffle = False, num_workers = 4, pin_memory = True)

    img, label = dataset[0]
    print(type(img))

    # -------------------------------
    # Saving model
    # -------------------------------

    epochs = args.epochs
    accuracy = np.zeros(len(epochs))
    trainer = pl.Trainer(default_root_dir=log_path, callbacks = [], deterministic=True, devices="auto", accelerator="auto")

    # -------------------------------
    # Testing model
    # -------------------------------

    _model = ExtendedCORnet(out_features = -1)

    for i, epoch in enumerate(epochs):
        model  = ExtendedCORnet(out_features = n_max)
        checkpoint = torch.load(f'{model_path}/epoch{epoch:02}.ckpt')
        model.load_state_dict(checkpoint['state_dict'])
        model.model.decoder.linear = _model.model.decoder.linear
        metrics, = trainer.test(model, test_loader)
        accuracy[i] = metrics['test_acc_epoch']
    """
    epoch_accuracy = np.zeros(len(epochs), dtype = np.dtype([('epoch', np.int64, 1), ('accuracy', np.float64, 1)]))
    epoch_accuracy['epoch'] = epochs
    epoch_accuracy['accuracy'] = accuracy
    os.makedirs(f'{accuracy_path}', exist_ok=True)
    np.save(f'{accuracy_path}/epochs_{epochs}_accuracy_imagenet.npy', epoch_accuracy)
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test models at different epochs')
    parser.add_argument('--epochs', metavar = 'E', type = int, nargs = "+", help = 'list of epochs to test')
    args = parser.parse_args()
    main(args)