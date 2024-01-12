import os
from typing import OrderedDict
import yaml
from pprint import pprint
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import logging
import torch
import pandas as pd
import numpy as np
import wandb

from collections import OrderedDict
from sacred import Experiment
from gpn.utils import RunConfiguration, DataConfiguration
from gpn.utils import ModelConfiguration, TrainingConfiguration
from gpn.experiments import MultipleRunExperiment
import warnings
from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

warnings.filterwarnings("ignore")

ex = Experiment("my_exp")

@ex.config
def config():
    # pylint: disable=missing-function-docstring
    overwrite = None
    db_collection = None


@ex.automain
def run_experiment(run: dict, data: dict, model: dict, training: dict):
    """main function to run experiment with sacred support

    Args:
        run (dict): configuration parameters of the job to run
        data (dict): configuration parameters of the data
        model (dict): configuration parameters of the model
        training (dict): configuration paramterers of the training

    Returns:
        dict: numerical results of the evaluation metrics for different splits
    """
    curr_dir = os.getcwd()
    model['curr_dir'] = curr_dir # for passing into gpn_base

    # home_dir = os.path.expanduser("~")
    # os.chdir(f"{home_dir}/Graph-Posterior-Network")

    run_cfg = RunConfiguration(**run)
    data_cfg = DataConfiguration(**data)
    model_cfg = ModelConfiguration(**model)
    train_cfg = TrainingConfiguration(**training)
    if torch.cuda.device_count() <= 0:
        run_cfg.set_values(gpu=False)
    else:
        print(torch.cuda.set_device(run_cfg.gpu))

    logging.info('Received the following configuration:')
    logging.info('RUN')
    logging.info(run_cfg.to_dict())
    logging.info('-----------------------------------------')
    logging.info('DATA')
    logging.info(data_cfg.to_dict())
    logging.info('-----------------------------------------')
    logging.info('MODEL')
    logging.info(model_cfg.to_dict())
    logging.info('-----------------------------------------')
    logging.info('TRAINING')
    logging.info(train_cfg.to_dict())
    logging.info('-----------------------------------------')

    wandb.init(project=run_cfg.experiment_name)

    experiment = MultipleRunExperiment(run_cfg, data_cfg, model_cfg, train_cfg, ex=ex)
    
    results = experiment.run()

    print(results.keys())
    metrics = [m[4:] for m in results.keys() if m.startswith('val_') and not m.endswith('_val')]
    result_values = {'val': [], 'test': []}
    
    for s in ('val', 'test'):
        for m in metrics:
            key = f'{s}_{m}'
            if key in results:
                val = results[key]
                if isinstance(val, list):
                    val = np.mean(val)
                result_values[s].append(val)
            else:
                result_values[s].append(None)

    df = pd.DataFrame(data=result_values, index=metrics)
    save_dir = os.path.join(run_cfg.experiment_directory, run_cfg.experiment_name, f"{data_cfg.dataset}_results.csv")
    df.to_csv(save_dir)
    print(df.to_markdown())

    # pprint(results)
    # return results

# if __name__ == '__main__':
#     warnings.filterwarnings("ignore")
#     config_path = "ood_loc_gpn_16"
#     # config_path = "classification_gpn_16"
#     with open(f'configs/gpn/{config_path}.yaml', 'r') as config_file:
#         config_updates = yaml.safe_load(config_file)

#     config_updates['data']['dataset'] = "Cora"

#     ex.run(config_updates = config_updates)
#     # ex.run_commandline()