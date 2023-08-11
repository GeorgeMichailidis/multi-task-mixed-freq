"""
Running multiple univariate benchmark methods on simulated data
(Copyright, 2023) Jiahe Lin & Michailidis

Suppoted methods: Naive, SES, ARIMA

To run:
python run_simUni.py --config=$config_filename --train_size=1000 --verbose=10
"""

import sys
import yaml
import argparse
import os
from datetime import datetime
import importlib

from helpers import SimEnvConfigurator, Evaluator

parser = argparse.ArgumentParser(description='train model on mixed frequency data')
parser.add_argument('--config', default='configs/ss00/seq2one.yaml')
parser.add_argument('--train_size', default=1000)
parser.add_argument('--verbose', default=10)
parser.add_argument('--output_folder_override', default='',type=str)

def main():

    """ main function for running simulation and record evaluation metrics"""
    raw_args = parser.parse_args()
    
    env_configurator = SimEnvConfigurator(raw_args, data_folder = 'data_sim', output_meta_folder = 'output_sim')
    env_configurator.config_directory_and_add_to_args()
    
    args = env_configurator.config_args(pickle_args=True)
    all_trainers = importlib.import_module('helpers')
    trainer_class = getattr(all_trainers, f'{args.model_type.lower()}SimTrainer')
    
    trainer = trainer_class(args=args,evaluator=Evaluator())
    trainer.source_data()
        
    if args.model_type == 'ARIMA':
        trainer.generate_train_val_datasets()
        trainer.config_and_train_model()
        
    trainer.run_forecast(pickle_predictions=True)
    trainer.eval_forecast(export_to_excel=True)
    
    with open(f'{args.output_folder}/args.txt', 'w') as f:
        print(vars(args), file=f)
    
if __name__ == "__main__":

    print('=============================================================')
    print(f'>>> {sys.argv[0]} started execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'>>> {sys.argv[0]} finished execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')
