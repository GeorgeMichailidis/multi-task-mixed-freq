"""
Running various methods on mixed-frequency real data
(Copyright, 2023) Jiahe Lin & Michailidis

Supported models: seq2seq, seq2one, transformer, gbm, mlp

To run:
python run_real.py --config=$config_filename --mode=static --verbose=1 --output_folder_override=''

"""

import sys
import yaml
import argparse
import os
from datetime import datetime

import pandas as pd
import tensorflow as tf

from helpers import EnvConfigurator
from helpers import nnTrainer, gbmTrainer

parser = argparse.ArgumentParser(description='train model on mixed frequency data')
parser.add_argument('--config', default='configs/FRED/seq2seq.yaml')
parser.add_argument('--mode', default='static', type=str, help='whether the model is dynamically retrained')
parser.add_argument('--verbose', default=1,type=int, help='verbose interval')
parser.add_argument('--output_folder_override', default='',type=str, help='override for output_folder; leave blank if default is used')

def main():

    """ main function for running simulation and record evaluation metrics"""
    raw_args = parser.parse_args()
    setattr(raw_args, 'verbose', int(raw_args.verbose))
    
    dataset_name = raw_args.config.split('/')[1]
    
    env_configurator = EnvConfigurator(raw_args,
                                       data_folder = f'data_{dataset_name}',
                                       output_meta_folder = f'output_{dataset_name}')
                                       
    env_configurator.config_directory_and_add_to_args()
    args = env_configurator.config_args(pickle_args=True)
    
    with open(f'{args.output_folder}/args.txt', 'w') as f:
        print(vars(args), file=f)
    
    if args.model_type != 'GBM':
        trainer = nnTrainer(args = args,criterion = tf.keras.losses.MeanSquaredError(),seed=411)
    else:
        trainer = gbmTrainer(args = args, seed_x=227, seed_y=228)
    
    trainer.set_seed()
    trainer.source_data()
    trainer.run_forecast()
    
if __name__ == "__main__":

    print('=============================================================')
    print(f'>>> {sys.argv[0]} started execution on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'>>> {sys.argv[0]} finished execution on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')
