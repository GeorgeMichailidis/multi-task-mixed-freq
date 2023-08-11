"""
Running various methods on simulated mixed-frequency data
(Copyright, 2023) Jiahe Lin & Michailidis

Supported methods:
- (multi-task-multi-step-ahead): MTMFSeq2Seq, transformer
- (multi-task-one-step-ahead): MTMFSeq2One
- (benchmarks): MLP, GBM

To run:
python run_sim.py --config=$config_filename --train_size=1000 --verbose=50
"""

import sys
import yaml
import argparse
import os
from datetime import datetime

import tensorflow as tf

from helpers import SimEnvConfigurator, Evaluator
from helpers import nnSimTrainer, gbmSimTrainer

parser = argparse.ArgumentParser(description='train model on mixed frequency data')
parser.add_argument('--config', default='configs/ss00/seq2one.yaml')
parser.add_argument('--train_size', default=1000)
parser.add_argument('--verbose', default=50,type=int)
parser.add_argument('--use_ckpt', default=0,type=int) ## boolean
parser.add_argument('--output_folder_override', default='',type=str)

def main():
    
    raw_args = parser.parse_args()
    
    env_configurator = SimEnvConfigurator(raw_args, data_folder = 'data_sim', output_meta_folder = 'output_sim')
    env_configurator.config_directory_and_add_to_args()
    
    global args
    args = env_configurator.config_args(pickle_args=True)
            
    if args.model_type != 'GBM':
        trainer = nnSimTrainer(args=args,
                               evaluator=Evaluator(),
                               criterion=tf.keras.losses.MeanSquaredError(),
                               seed=None)
    else:
        trainer = gbmSimTrainer(args=args, evaluator=Evaluator())
        
    trainer.set_seed(repickle_args=True)
    trainer.source_data()
    trainer.generate_train_val_datasets()
    trainer.config_and_train_model()
    trainer.eval_training(print_train_err=True,plot_fitted=True)
    trainer.config_predictor()
    
    trainer.run_forecast(pickle_predictions=True)
    trainer.eval_forecast(export_to_excel=True)
    
    with open(f'{args.output_folder}/args.txt', 'w') as f:
        print(vars(args), file=f)
    
if __name__ == "__main__":

    print('=============================================================')
    print(f'>>> tf version: {tf.__version__}; available GPU devices:')
    print(tf.config.list_physical_devices('GPU'))
    print(f'>>> {sys.argv[0]} started execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'>>> {sys.argv[0]} finished execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')
