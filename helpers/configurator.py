"""
Classes for configuring the evnrionment, including loading configs, creating directories etc.
"""

import sys
import yaml
import os
import shutil
import pickle
import pandas as pd
import numpy as np

class SimEnvConfigurator():

    def __init__(self, args, data_folder = 'data_sim', output_meta_folder = 'output_sim'):
        """
        initialization
        args = parser.parse_args() comes from parsing cmd line input
        """
        self.data_folder = data_folder
        self.output_meta_folder = output_meta_folder
        
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if config.get('includes') is not None:
            for item in config.get('includes'):
                with open(item) as handle:
                    config_this_item = yaml.safe_load(handle)
                for key in config_this_item:
                    for k, v in config_this_item[key].items():
                        setattr(args, k, v)
            del config['includes']
        
        if config['setup']['model_type'] not in ['GBM','DeepAR','NHiTS','ARIMA']:
            for key in config:
                for k, v in config[key].items():
                    setattr(args, k, v)
        else:
            for key in config:
                if key in ['hyper_params','hyper_params_x', 'hyper_params_y']:
                    setattr(args, key, config[key])
                else:
                    for k, v in config[key].items():
                        setattr(args, k, v)
                    
        ## ensure cmd line input is of the correct type
        setattr(args,'train_size',int(args.train_size))
        if hasattr(args, 'use_ckpt'):
            setattr(args,'use_ckpt',int(args.use_ckpt))
        else:
            setattr(args,'use_ckpt',0)
        if hasattr(args, 'verbose'):
            setattr(args,'verbose',int(args.verbose))
        
        self.args = args
    
    def config_directory_and_add_to_args(self, delete_existing=False):
    
        setattr(self.args,'data_folder',self.data_folder)
        
        setattr(self.args,'output_parent_folder', f"{self.output_meta_folder}/{self.args.ds_name}")
        if hasattr(self.args, 'output_folder_override') and len(self.args.output_folder_override):
            setattr(self.args,'output_folder', f"{self.args.output_parent_folder}/{self.args.output_folder_override}")
        else:
            setattr(self.args,'output_folder', f"{self.args.output_parent_folder}/{self.args.model_type}_{self.args.train_size}")
        
        if self.args.use_ckpt:
            setattr(self.args,'ckpt_folder',f'{self.args.output_folder}/ckpt')
            
        if not os.path.exists(self.args.output_folder):
            os.makedirs(self.args.output_folder)
            print(f'folder {self.args.output_folder}/ created')
        else:
            if delete_existing:
                print(f'folder {self.args.output_folder}/ exists; deleted and recreated to ensure there is no stale output for this run')
                shutil.rmtree(self.args.output_folder)
                os.mkdir(self.args.output_folder)
            else:
                print(f'folder {self.args.output_folder}/ exists; no action needed')
                pass
            
    def config_args(self, pickle_args = True):
    
        x = pd.read_excel(f"{self.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='x')
        y = pd.read_excel(f"{self.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='y')
        xdata, ydata = x.values, y.values
        
        ## set dimension
        setattr(self.args, 'dim_x', xdata.shape[1])
        setattr(self.args, 'dim_y', ydata.shape[1])
    
        ## setup for benchmark models where the output for high freq is not a sequence
        if self.args.model_type in ['MTMFSeq2One','MLP','RNN','GBM']:
            setattr(self.args, 'Tx', 1)
            
        if self.args.model_type in ['MLP','GBM','DeepAR','NHiTS','ARIMA']:
            setattr(self.args, 'zero_pad', False)
        else:
            setattr(self.args, 'zero_pad', True)
            
        if pickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        return self.args

class EnvConfigurator():
    def __init__(self, args, data_folder = 'data_sim', output_meta_folder = 'output_sim'):
        """
        initialization
        args = parser.parse_args() comes from parsing cmd line input
        """
        self.data_folder = data_folder
        self.output_meta_folder = output_meta_folder
        
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
        if config.get('includes') is not None:
            for item in config.get('includes'):
                with open(item) as handle:
                    config_this_item = yaml.safe_load(handle)
                for key in config_this_item:
                    for k, v in config_this_item[key].items():
                        setattr(args, k, v)
            del config['includes']
        
        if config['setup']['model_type'] not in ['GBM','DeepAR','NHiTS','ARIMA']:
            for key in config:
                for k, v in config[key].items():
                    setattr(args, k, v)
        else:
            for key in config:
                if key in ['hyper_params','hyper_params_x', 'hyper_params_y']:
                    setattr(args, key, config[key])
                else:
                    for k, v in config[key].items():
                        setattr(args, k, v)
                        
        setattr(args,'first_prediction_date',pd.to_datetime(args.first_prediction_date))
        
        ## ensure cmd line input is of the correct type
        if hasattr(args,'verbose'):
            setattr(args,'verbose',int(args.verbose))
        
        self.args = args
    
    def config_directory_and_add_to_args(self, delete_existing=False):
    
        setattr(self.args,'data_folder',self.data_folder)
        if hasattr(self.args, 'output_folder_override') and len(self.args.output_folder_override):
            setattr(self.args,'output_folder', f"{self.output_meta_folder}/{self.args.output_folder_override}")
        else:
            setattr(self.args,'output_folder', f"{self.output_meta_folder}/{self.args.model_type}_{self.args.mode}")
        setattr(self.args,'output_filename', f"{self.args.output_folder}/predictions.xlsx")
            
        if not os.path.exists(self.args.output_folder):
            os.makedirs(self.args.output_folder)
            print(f'folder {self.args.output_folder}/ created')
        else:
            if delete_existing:
                print(f'folder {self.args.output_folder}/ exists; deleted and recreated to ensure there is no stale output for this run')
                shutil.rmtree(self.args.output_folder)
                os.mkdir(self.args.output_folder)
            else:
                print(f'folder {self.args.output_folder}/ exists; no action needed')
                pass
                
    def config_args(self, pickle_args = True):
    
        x = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='x')
        y = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='y')
        xdata, ydata = x.values, y.values
        
        ## set dimension
        setattr(self.args, 'dim_x', xdata.shape[1])
        setattr(self.args, 'dim_y', ydata.shape[1])
    
        ## setup for benchmark models where the output for high freq is not a sequence
        if self.args.model_type in ['MTMFSeq2One','MLP','RNN','GBM']:
            setattr(self.args, 'Tx', 1)
            
        if self.args.model_type in ['MLP','GBM','DeepAR','NHiTS','ARIMA']:
            setattr(self.args, 'zero_pad', False)
        else:
            setattr(self.args, 'zero_pad', True)
        
        if pickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        return self.args
