import json
import os
import sys

import torch

from dataloader.dataloader import get_data
from models.models import AutoEncoder, UNet, UNetPlus
from models.trainer import Trainer

from utility.utility import create_folders, set_reproducibility, define_structure
from utility.test import test


def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    gpu = params["n_gpu"]
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    params["device"] = device

    model_name = params["model_name"]
    dataset_name = params["dataset_name"]

    create_folders(params)
    seed = params["seed"]
    set_reproducibility(seed)

    exp = f"Sanitization_{dataset_name}_with_{model_name}"
    params = define_structure(params, exp)

    # Get data
    print("[INFO] loading dataset...")
    (train_loader, val_loader, test_loader_classic_rows_loader, test_loader_classic_squares_loader,
     test_loader_classic_sequential_loader, test_loader_oceanlotus_rows_loader, test_loader_oceanlotus_squares_loader,
     test_loader_oceanlotus_sequential_loader) = get_data(params)

    idx = params['MODEL_PATH'].rfind('.')
    mod = __import__(params['MODEL_PATH'][:idx], fromlist=params['MODEL_PATH'][idx + 1:])
    Model = getattr(mod, params['MODEL_PATH'][idx + 1:])

    model = Model(input_shape=params['n_channels'], init_kernel_size = (params['init_ks_1'],
                                                                                  params['init_ks_2']),
                            init_stride=params['init_stride'],
                            init_padding = (params['init_padding1'], params['init_padding1']),
                            output_padding=params['output_padding'])

    if params['train']:
        trainer = Trainer(model, params)
        trainer.train(train_loader, val_loader, batch_size=params['batch_size'], num_epochs=params['n_epochs'])

    if params['test']:
        if params['load_best_model']:
            model.load_state_dict(torch.load(params['best_path_model']))
        else:
            model.load_state_dict(torch.load(params['last_path_model']))

        # LSB CLASSIC
        params['TYPE_DIR'] = 'test/lsb_classic/interleaving_rows'
        test(test_loader_classic_rows_loader, model, params)

        params['TYPE_DIR'] = 'test/lsb_classic/interleaving_squares'
        test(test_loader_classic_squares_loader, model, params)

        params['TYPE_DIR'] = 'test/lsb_classic/sequential'
        test(test_loader_classic_sequential_loader, model, params)

        # LSB OCEANLOTUS
        params['TYPE_DIR'] = 'test/lsb_oceanlotus/interleaving_rows'
        test(test_loader_oceanlotus_rows_loader, model, params)

        params['TYPE_DIR'] = 'test/lsb_oceanlotus/interleaving_squares'
        test(test_loader_oceanlotus_squares_loader, model, params)

        params['TYPE_DIR'] = 'test/lsb_oceanlotus/sequential'
        test(test_loader_oceanlotus_sequential_loader, model, params)



