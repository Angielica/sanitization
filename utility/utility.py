import numpy as np
import random
import torch
import os
def set_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

def create_folders(params):
    SAVE_FOLDER, MODEL_FOLDER, PLOT_FOLDER = params['SAVE_FOLDER'], params['MODEL_FOLDER'], params['PLOT_FOLDER']
    RESULT_FOLDER = params['RESULT_FOLDER']
    OUT_DIR = params['OUT_DIR']

    for dir in [SAVE_FOLDER, MODEL_FOLDER, PLOT_FOLDER, RESULT_FOLDER, OUT_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)

def define_structure(params, exp):
    SAVE_FOLDER, MODEL_FOLDER, PLOT_FOLDER= params['SAVE_FOLDER'], params['MODEL_FOLDER'], params['PLOT_FOLDER']

    best_path_model, last_path_model = f'best_model_{exp}.pt', f'last_model_{exp}.pt'
    params['best_path_model'] = os.path.join(MODEL_FOLDER, best_path_model)
    params['last_path_model'] = os.path.join(MODEL_FOLDER, last_path_model)

    path_training_loss, path_training_loss_log = f'training_loss_{exp}', f'training_log_loss_{exp}'
    params['path_training_loss'] = os.path.join(PLOT_FOLDER, path_training_loss)
    params['path_training_loss_log'] = os.path.join(PLOT_FOLDER, path_training_loss_log)

    return params