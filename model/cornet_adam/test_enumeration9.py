    import torch
import pytorch_lightning as pl
from nn_modeling.model.torch import ExtendedCORnet
from nn_modeling.dataset.enumeration import CountingDotsDataSet
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

    # path where activities are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/enumeration_{n_max}/activity/cornet_adam'
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

    def hook_output(m, i, o):
        activity[m].append(o.cpu())

    class LabelConditionCallback(Callback):
        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            label.append(batch[1].cpu())
            condition.append(batch[2].cpu())
            param.append(batch[3].cpu())

    epochs = args.epochs
    accuracy = np.zeros(len(epochs))
    trainer = pl.Trainer(default_root_dir=log_path, callbacks = [LabelConditionCallback()], deterministic=True, devices="auto", accelerator="auto")
    model = ExtendedCORnet(out_features = n_max)

    modules = [getattr(model.model, m).output for m in ["V1", "V2", "V4", "IT", "decoder"]]
    module_names = {getattr(model.model, m).output:m for m in ["V1", "V2", "V4", "IT", "decoder"]}
    times = {getattr(model.model, m).output:getattr(model.model, m).times if hasattr(getattr(model.model, m), "times") else 1 for m in ["V1", "V2", "V4", "IT", "decoder"]}
    for m in modules:
        m.register_forward_hook(hook_output)

    # -------------------------------
    # Testing model
    # -------------------------------

    for i, epoch in enumerate(epochs):
        os.makedirs(f'{activity_path}/epoch{epoch:02}', exist_ok=True)
        checkpoint = torch.load(f'{model_path}/epoch{epoch:02}.ckpt')
        model.load_state_dict(checkpoint['state_dict'])
        activity = {}
        label = []
        condition = []
        param = []
        for m in modules:
            activity[m]= []
        metrics, = trainer.test(model, test_loader)
        label = torch.cat(label).cpu().numpy()[:, None]
        if not os.path.exists(f'{activity_path}/epoch{epoch:02}/label.npz'):
            np.savez_compressed(f'{activity_path}/epoch{epoch:02}/label.npz', label = label)
            print(f'label at epoch {epoch} saved in .npz ({format_size(process.memory_info().rss)})')
        else:
            print(f'label at epoch {epoch} already saved in .npz ({format_size(process.memory_info().rss)})')
        condition = torch.cat(condition).cpu().numpy()[:, None]
        if not os.path.exists(f'{activity_path}/epoch{epoch:02}/condition.npz'):
            np.savez_compressed(f'{activity_path}/epoch{epoch:02}/condition.npz', condition = condition)
            print(f'condition at epoch {epoch} saved in .npz ({format_size(process.memory_info().rss)})')
        else:
            print(f'condition at epoch {epoch} already saved in .npz ({format_size(process.memory_info().rss)})')
        param = torch.cat(param).cpu().numpy()[:, None]
        if not os.path.exists(f'{activity_path}/epoch{epoch:02}/param.npz'):
            np.savez_compressed(f'{activity_path}/epoch{epoch:02}/param.npz', param = param)
            print(f'param at epoch {epoch} saved in .npz ({format_size(process.memory_info().rss)})')
        else:
            print(f'param at epoch {epoch} already saved in .npz ({format_size(process.memory_info().rss)})')
        print(f'Test finished ({format_size(process.memory_info().rss)})')
        accuracy[i] = metrics['test_acc_epoch']
        for m in modules:
            if not os.path.exists(f'{activity_path}/epoch{epoch:02}/{module_names[m]}.npz'):
                print(f'starting saving {module_names[m]} at epoch {epoch} ({format_size(process.memory_info().rss)})')
                tmp = torch.stack([torch.cat(activity[m][i::times[m]]) for i in range(times[m])], axis = 1).numpy()
                print(f'tmp created ({format_size(process.memory_info().rss)})')
                del activity[m]
                print(f'activity[m] removed ({format_size(process.memory_info().rss)})')
                print(module_names[m], tmp.shape)
                data_dict = {}
                data_dict[module_names[m]] = tmp
                print(f'data_dict created ({format_size(process.memory_info().rss)})')
                np.savez_compressed(f'{activity_path}/epoch{epoch:02}/{module_names[m]}.npz', **data_dict)
                print(f'{module_names[m]} at epoch {epoch} saved in .npz ({format_size(process.memory_info().rss)})')
                del tmp
                del data_dict
            else:
                print(f'{module_names[m]} at epoch {epoch} already saved in .npz ({format_size(process.memory_info().rss)})')

    epoch_accuracy = np.zeros(len(epochs), dtype = np.dtype([('epoch', np.int64, 1), ('accuracy', np.float64, 1)]))
    epoch_accuracy['epoch'] = epochs
    epoch_accuracy['accuracy'] = accuracy
    os.makedirs(f'{accuracy_path}', exist_ok=True)
    np.save(f'{accuracy_path}/epochs_{epochs}_accuracy.npy', epoch_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test models at different epochs')
    parser.add_argument('--epochs', metavar = 'E', type = int, nargs = "+", help = 'list of epochs to test')
    args = parser.parse_args()
    main(args)