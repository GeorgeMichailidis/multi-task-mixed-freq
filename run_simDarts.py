"""
Running benchmark methods on simulated data, leveraging directly the implementations in darts
(Copyright, 2023) Jiahe Lin & Michailidis

Supported methods: DeepAR, NHiTS

To run:
python run_simDarts.py --config=$config_filename --train_size=1000 --verbose=50
"""
import sys
import yaml
import argparse
import os
from datetime import datetime

import torch
from helpers import SimEnvConfigurator, Evaluator
from helpers import deepARSimTrainer, nhitsSimTrainer

parser = argparse.ArgumentParser(description='train model on mixed frequency data')
parser.add_argument('--config', default='configs/ss00/deepar.yaml')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--train_size', type=int, default=1000)
parser.add_argument('--output_folder_override', default='',type=str)

def main():

    """ main function for running simulation and record evaluation metrics"""
    raw_args = parser.parse_args()
    
    env_configurator = SimEnvConfigurator(raw_args, data_folder = 'data_sim', output_meta_folder = 'output_sim')
    env_configurator.config_directory_and_add_to_args()
    
    args = env_configurator.config_args(pickle_args=True)
            
    if args.model_type == 'DeepAR':
        trainer = deepARSimTrainer(args=args,
                                   evaluator=Evaluator())
    elif args.model_type == 'NHiTS':
        trainer = nhitsSimTrainer(args=args,
                                  evaluator=Evaluator())
    else:
        raise ValueError('unsupported model_type')
        
    trainer.set_seed(repickle_args=True)
    trainer.source_data()
    trainer.generate_train_val_datasets()
    trainer.config_and_train_model()
    trainer.run_forecast(pickle_predictions=True)
    trainer.eval_forecast(export_to_excel=True)
    
    #try:
    #    print(f'attempting to generating plots for training/val evaluation ...')
    trainer.eval_training()
    #except:
    #    pass
    
    with open(f'{args.output_folder}/args.txt', 'w') as f:
        print(vars(args), file=f)
    
if __name__ == "__main__":

    print('=============================================================')
    print(f'>>> torch version: {torch.__version__}; available GPU devices:')
    print([torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())])
    print(f'>>> {sys.argv[0]} started execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'>>> {sys.argv[0]} finished execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')
