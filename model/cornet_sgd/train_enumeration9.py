import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nn_modeling.dataset.enumeration import CountingDotsDataSet
from nn_modeling.model.torch import ExtendedCORnet
import numpy as np
from functools import reduce

seed = 0
pl.seed_everything(seed)

# -------------------------------
# Parameters
# -------------------------------

n_max = 9 # max number of dots
n_sample = 10 # number of samples used in training per class/condition
sample = np.arange(n_sample) # id of sample used
n_epoch = 50 # number of epochs max used to train

# -------------------------------
# Paths where to load/save data
# -------------------------------

# path where model are saved
model_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/model/cornet_sgd'
os.makedirs(f'{model_path}', exist_ok=True)
# path where log of training are saved
log_path = f'{os.environ.get("TMP_PATH")}/enumeration_{n_max}/log/train'
# path containing the dataset
dataset_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/stimuli'

# -------------------------------
# Training dataset
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
train_loader = DataLoader(dataset, batch_size = 50, shuffle = True, num_workers = 4, pin_memory = True)

# -------------------------------
# Initializing model
# -------------------------------

model  = ExtendedCORnet(out_features = n_max, lr = 1e-3, optimizer = 'sgd')

# -------------------------------
# Saving model
# -------------------------------

# saving initial model
torch.save({
	"epoch": -1,
	"global_step": 0,
	"pytorch-lightning_version": pl.__version__,
	"state_dict": model.state_dict()
}, f'{model_path}/epoch-1.ckpt')
# using checkpoint to save models after each epoch
checkpoint = pl.callbacks.ModelCheckpoint(dirpath = model_path, filename = 'epoch{epoch:02d}', auto_insert_metric_name = False, save_on_train_epoch_end = True, save_top_k = -1)
# saving gpu stats
gpu_stats = pl.callbacks.DeviceStatsMonitor()

# -------------------------------
# Training model
# -------------------------------

trainer = pl.Trainer(default_root_dir = log_path, callbacks = [gpu_stats, checkpoint], deterministic = True, accelerator='gpu', devices=4, strategy = "ddp", num_nodes = 1, max_epochs = 50)
trainer.fit(model, train_loader)