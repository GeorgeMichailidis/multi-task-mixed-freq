"""
Trainer class for running gbm on real data
"""

import sys
import os
import shutil
import pickle

import pandas as pd
import numpy as np
from datetime import datetime

from helpers import ClsConstructor
from .utils_train import *

class gbmTrainer():

    def __init__(self,args,seed_x = None, seed_y = None):
        
        self.args = args
        self.cls_constructor = ClsConstructor(self.args)
        self.seed_x = seed_x or 227
        self.seed_y = seed_y or 228
    
    def set_seed(self, repickle_args=True):
        
        self.args.hyper_params_x['seed'] = self.seed_x
        self.args.hyper_params_y['seed'] = self.seed_y
        
        if repickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    def source_data(self):
        
        x = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='x')
        y = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='y')
        x.index, y.index = pd.to_datetime(x.index), pd.to_datetime(y.index)
        xdata, ydata = x.values, y.values
        
        self.df_info = {'x_index': list(x.index),
                        'y_index': list(y.index),
                        'x_columns': list(x.columns),
                        'y_columns': list(y.columns),
                        'x_total_obs': x.shape[0],
                        'y_total_obs': y.shape[0]
                       }
        self.raw_data = (xdata, ydata)
        
    def generate_train_val_datasets(self, x_train_end, y_train_end, n_val = None):
        
        """
        helper function for generating train/val dataset; the reason for not adding them as attributes is out of
        consideration for dynamic run
        Argv:
        - x_train_end, y_train_end: the ending index, resp for x and y
        """
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        
        ## get training data
        x_train, y_train = xdata[:x_train_end,:], ydata[:y_train_end,:]
        train_inputs, train_targets = dp.mf_sample_generator(x_train,
                                                             y_train,
                                                             update_scaler = True if args.scale_data else False,
                                                             apply_scaler = True if args.scale_data else False)
        
        print(f'[train samples generated] inputs dims: {[temp.shape for temp in train_inputs]}; target dims: {[temp.shape for temp in train_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        if n_val is not None:
            if isinstance(n_val, float):
                n_val = round(train_inputs[0].shape[0] * n_val)
            
            train_inputs, val_inputs = [x[:-n_val] for x in train_inputs], [x[-n_val:] for x in train_inputs]
            train_targets, val_targets = [x[:-n_val] for x in train_targets], [x[-n_val:] for x in train_targets]
            print(f'[{n_val} val samples reserved] inputs dims: {[temp.shape for temp in val_inputs]}; target dims: {[temp.shape for temp in val_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        train_data = (train_inputs, train_targets)
        val_data = (val_inputs, val_targets) if n_val is not None else None
        
        return dp, train_data, val_data
        
    def config_and_train_model(self, train_data, val_data = None):
        
        args = self.args
        model = self.cls_constructor.create_model()
        
        ## compile
        print(f'[{args.model_type} model training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        history = model.train(train_dataset = train_data, val_dataset = val_data, verbose=args.verbose)
        
        return model
            
    def eval_train(self, model, dp, train_data):
        
        args = self.args
        train_inputs, train_targets = train_data
        
        x_is_pred, y_is_pred = model.predict(train_inputs)
        if args.scale_data:
            x_is_pred = dp.apply_scaler('scaler_x', x_is_pred, inverse=True)
            y_is_pred = dp.apply_scaler('scaler_y', y_is_pred, inverse=True)
            x_is_truth = dp.apply_scaler('scaler_x', train_targets[0], inverse=True)
            y_is_truth = dp.apply_scaler('scaler_y', train_targets[1], inverse=True)
        else:
            x_is_truth, y_is_truth = train_targets[0], train_targets[1]
            
        plot_fitted_val(args, (x_is_pred, y_is_pred), (x_is_truth, y_is_truth), x_col = self.df_info['x_columns'].index(args.X_COLNAME), y_col = self.df_info['y_columns'].index(args.Y_COLNAME), time_steps = None, save_as_file = f'{args.output_folder}/train_fit_static.png')
    
    def config_predictor(self, model, dp):
        
        args = self.args
        predictor = self.cls_constructor.create_predictor(model, dp, apply_inv_scaler = args.scale_data)
        
        return predictor
    
    def run_forecast_one_set(self, predictor, dp, y_start_id, x_start_id):
        """ helper function for running one set of forecast: F, N1, N2, N3 """
        args = self.args
        xdata, ydata = self.raw_data
        
        assert x_start_id == args.freq_ratio * (y_start_id - 1)
        predictions_by_vintage = {}
        
        for x_step in range(args.freq_ratio+1):
            experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
            inputs, targets = dp.create_one_forecast_sample(xdata,
                                                            ydata,
                                                            x_start_id,
                                                            y_start_id,
                                                            x_step = x_step,
                                                            horizon = args.horizon ,
                                                            apply_scaler = True if args.scale_data else False,
                                                            verbose= False)
                                                            
            x_pred, y_pred = predictor(inputs, x_step = x_step, horizon = args.horizon)
            predictions_by_vintage[experiment_tag] = {'x_pred': x_pred, 'y_pred': y_pred}
        return predictions_by_vintage
    
    def add_prediction_to_collector(self, predictions_by_vintage, T_datestamp, x_PRED_collector = [], y_PRED_collector = []):
        
        args = self.args
        
        ## initialization for recording the forecast
        x_numeric_col_keys = [f'step_{i+1}' for i in range(args.freq_ratio * args.horizon)]
        y_numeric_col_keys = [f'step_{i+1}' for i in range(args.horizon)]
        
        ## extract prediction
        for vintage, predictions in predictions_by_vintage.items():
            x_pred, y_pred = predictions['x_pred'], predictions['y_pred']
            for col_id, variable_name in enumerate(self.df_info['x_columns']):
                temp = {'prev_QE': T_datestamp, 'tag': vintage, 'variable_name': variable_name}
                temp.update(dict(zip(x_numeric_col_keys, list(x_pred[:,col_id]))))
                x_PRED_collector.append(temp)
            for col_id, variable_name in enumerate(self.df_info['y_columns']):
                temp = {'prev_QE': T_datestamp, 'tag': vintage, 'variable_name': variable_name}
                temp.update(dict(zip(y_numeric_col_keys, list(y_pred[:,col_id]))))
                y_PRED_collector.append(temp)
        
    def run_forecast(self):
    
        """ main function """
    
        args = self.args
        ## initialization for recording the forecast
        x_PRED_collector, y_PRED_collector = [], []
        
        if args.mode == 'static':
            
            x_train_end = self.df_info['x_index'].index(args.first_prediction_date) - args.freq_ratio + 1 ## +1 to ensure the index is inclusive
            y_train_end = self.df_info['y_index'].index(args.first_prediction_date) - 2 + 1 ## +1 to ensure the index is inclusive
            
            ## set up and train model
            dp, train_data, val_data = self.generate_train_val_datasets(x_train_end, y_train_end, n_val = args.n_val)
            model = self.config_and_train_model(train_data, val_data=val_data)
            self.eval_train(model, dp, train_data)
            predictor = self.config_predictor(model, dp)
            
            ## run rolling forecast based on the trained model
            y_start_id = self.df_info['y_index'].index(args.first_prediction_date) - (args.Ty - 1)
            x_start_id = args.freq_ratio * (y_start_id - 1)
            test_size = self.df_info['y_total_obs'] - self.df_info['y_index'].index(args.first_prediction_date)
            
            print(f'[forecast starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            for experiment_id in range(test_size):
                
                T_datestamp = self.df_info['x_index'][x_start_id + args.Lx - 1]
                x_range = [self.df_info['x_index'][x_start_id], self.df_info['x_index'][x_start_id + args.Lx - 1]]
                y_range = [self.df_info['y_index'][y_start_id], self.df_info['y_index'][y_start_id + args.Ty - 2]]
                
                print(f" >> id = {experiment_id+1}/{test_size}: prev timestamp = {T_datestamp}; x_input_range = {x_range}, y_input_range = {y_range}")
                
                predictions_by_vintage = self.run_forecast_one_set(predictor, dp, y_start_id, x_start_id)
                self.add_prediction_to_collector(predictions_by_vintage, T_datestamp, x_PRED_collector, y_PRED_collector)
            
                # update x_start_id and y_start_id
                x_start_id += args.freq_ratio
                y_start_id += 1
            print(f'[forecast ends] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
        elif args.mode == 'dynamic':
        
            offset = self.df_info['y_index'].index(args.first_prediction_date)
            test_size = self.df_info['y_total_obs'] - offset
            
            for experiment_id in range(test_size):
                
                pred_date = self.df_info['y_index'][offset + experiment_id]
                
                print(f' >> id = {experiment_id+1}/{test_size}: next QE = {pred_date.strftime("%Y-%m-%d")}')
                ## set up and train model
                x_train_end = self.df_info['x_index'].index(pred_date) - args.freq_ratio + 1 ## +1 to ensure the index is inclusive
                y_train_end = self.df_info['y_index'].index(pred_date) - 2 + 1 ## +1 to ensure the index is inclusive
                dp, train_data, val_data = self.generate_train_val_datasets(x_train_end, y_train_end, n_val = args.n_val)
                model = self.config_and_train_model(train_data, val_data=val_data)
                self.eval_train(model, dp, train_data)
                predictor = self.config_predictor(model, dp)
                
                ## run forecast
                y_start_id = self.df_info['y_index'].index(pred_date) - (args.Ty - 1)
                x_start_id = args.freq_ratio * (y_start_id - 1)
                T_datestamp = self.df_info['x_index'][x_start_id + args.Lx - 1]
                
                predictions_by_vintage = self.run_forecast_one_set(predictor, dp, y_start_id, x_start_id)
                self.add_prediction_to_collector(predictions_by_vintage, T_datestamp, x_PRED_collector, y_PRED_collector)
                
                del predictor
                del model
                    
        x_PRED_df, y_PRED_df = pd.DataFrame(x_PRED_collector), pd.DataFrame(y_PRED_collector)
        
        with pd.ExcelWriter(args.output_filename) as writer:
            x_PRED_df.to_excel(writer,sheet_name=f'x_prediction',index=False)
            y_PRED_df.to_excel(writer,sheet_name=f'y_prediction',index=False)
