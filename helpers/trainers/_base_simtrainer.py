"""
base class for simulation train wrapper
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from .utils_train import plot_fitted_val, export_error_to_excel

class _baseSimTrainer():

    def __init__(self, args, evaluator):
        
        self.args = args
        self.evaluator = evaluator
        self.cls_constructor = None
        
    def source_data(self):
        
        x = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='x')
        y = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='y')
        
        self.raw_data = (x.values, y.values)
        self.data_cols = (x.columns, y.columns)
    
    def generate_train_val_datasets(self):
        
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        
        ## get training data
        x_train_len = dp.find_length(args.train_size, len_input = args.Lx, len_target = args.Tx, stride = args.freq_ratio)
        y_train_len = dp.find_length(args.train_size, len_input = args.Ty-1, len_target = 1, stride = 1)
        x_train, y_train = xdata[:x_train_len,:], ydata[:y_train_len,:]
        train_inputs, train_targets = dp.mf_sample_generator(x_train,
                                                             y_train,
                                                             update_scaler = True if args.scale_data else False,
                                                             apply_scaler = True if args.scale_data else False)

        ## get validation data
        x_val_len = dp.find_length(args.val_size, len_input = args.Lx, len_target = args.Tx, stride = args.freq_ratio)
        y_val_len = dp.find_length(args.val_size, len_input = args.Ty-1, len_target = 1, stride = 1)
        x_val_start, y_val_start = args.train_size * args.freq_ratio, args.train_size
        x_val_end, y_val_end = x_val_start + x_val_len, y_val_start + y_val_len
        x_val, y_val = xdata[x_val_start : x_val_end,:], ydata[y_val_start : y_val_end,:]
        val_inputs, val_targets = dp.mf_sample_generator(x_val,
                                                         y_val,
                                                         update_scaler = False,
                                                         apply_scaler = True if args.scale_data else False)
        print(f'[train samples generated] inputs dims: {[temp.shape for temp in train_inputs]}; target dims: {[temp.shape for temp in train_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        self.dp = dp
        self.train_dataset = (train_inputs, train_targets)
        self.val_dataset = (val_inputs, val_targets)
    
    def eval_training(self, print_train_err = True, plot_fitted = True):
        
        args = self.args
        model, dp, evaluator = self.model, self.dp, self.evaluator
        
        train_inputs, train_targets = self.train_dataset
        x_is_pred, y_is_pred = model.predict(train_inputs)
        if args.scale_data:
            x_is_pred = dp.apply_scaler('scaler_x', x_is_pred, inverse=True)
            y_is_pred = dp.apply_scaler('scaler_y', y_is_pred, inverse=True)
            x_is_truth = dp.apply_scaler('scaler_x', train_targets[0], inverse=True)
            y_is_truth = dp.apply_scaler('scaler_y', train_targets[1], inverse=True)
        else:
            x_is_truth, y_is_truth = train_targets[0], train_targets[1]
        
        val_inputs, val_targets = self.val_dataset
        x_val_pred, y_val_pred = model.predict(val_inputs)
        if args.scale_data:
            x_val_pred = dp.apply_scaler('scaler_x', x_val_pred, inverse=True)
            y_val_pred = dp.apply_scaler('scaler_y', y_val_pred, inverse=True)
            x_val_truth = dp.apply_scaler('scaler_x', val_targets[0], inverse=True)
            y_val_truth = dp.apply_scaler('scaler_y', val_targets[1], inverse=True)
        else:
            x_val_truth, y_val_truth = val_targets[0], val_targets[1]

        if args.model_type in ['MTMFSeq2Seq','transformer']:
            x_train_RMSE = evaluator.batch_eval('rmse_by_step', x_is_truth, x_is_pred)
            y_train_RMSE = evaluator.batch_eval('rmse', y_is_truth, y_is_pred)
            x_val_RMSE = evaluator.batch_eval('rmse_by_step', x_val_truth, x_val_pred)
            y_val_RMSE = evaluator.batch_eval('rmse', y_val_truth, y_val_pred)
        else:
            x_train_RMSE = evaluator.batch_eval('rmse', x_is_truth, x_is_pred)
            y_train_RMSE = evaluator.batch_eval('rmse', y_is_truth, y_is_pred)
            x_val_RMSE = evaluator.batch_eval('rmse',x_val_truth, x_val_pred)
            y_val_RMSE = evaluator.batch_eval('rmse',y_val_truth, y_val_pred)

        if print_train_err:
            with open(f'{args.output_folder}/train_val_err.txt', 'w') as f:
                f.write('RMSE when both the input and the target are inversely transformed back, averaged across all samples and coordinates\n')
                f.write(f'x_train_RMSE = {x_train_RMSE}; y_train_RMSE = {y_train_RMSE:.2f}\n')
                f.write(f'x_val_RMSE = {x_val_RMSE}; y_val_RMSE = {y_val_RMSE:.2f}\n')
        
        if plot_fitted:
            ## plot fitted value for both train and validation
            plot_fitted_val(args, (x_is_pred, y_is_pred), (x_is_truth, y_is_truth), save_as_file=f'{args.output_folder}/train_fit_rand_col.png')
            plot_fitted_val(args, (x_val_pred, y_val_pred), (x_val_truth, y_val_truth), save_as_file=f'{args.output_folder}/val_fit_rand_col.png')
    
    def eval_forecast(self, export_to_excel = True):
        
        x_pred_recorder, y_pred_recorder = self.x_pred_recorder, self.y_pred_recorder
        
        metric_fn = getattr(self.evaluator, self.args.eval_metric)
        experiment_tags = list(x_pred_recorder.keys())
        
        ## raw per-step-RMSE values for each experiment
        x_err_collect, y_err_collect = {}, {}
        for experiment_tag in experiment_tags:
            for target_pred in x_pred_recorder[experiment_tag]:
                x_err_curr = metric_fn(target_pred['target'],target_pred['prediction'])
                x_err_collect.setdefault(experiment_tag,[]).append(x_err_curr)
            x_err_collect[experiment_tag] = np.array(x_err_collect[experiment_tag])
            for target_pred in y_pred_recorder[experiment_tag]:
                y_err_curr = metric_fn(target_pred['target'],target_pred['prediction'])
                y_err_collect.setdefault(experiment_tag,[]).append(y_err_curr)
            y_err_collect[experiment_tag] = np.array(y_err_collect[experiment_tag])
            
        ## summary for x
        x_indices = [f'step_{i+1}' for i in range(self.args.horizon * self.args.freq_ratio)]
        x_err_median = pd.DataFrame(data = np.stack([np.median(err,axis=0) for err in x_err_collect.values()],axis=1),
                                    columns = experiment_tags, index = x_indices)
        x_err_median['metric'] = 'median'
        x_err_mean = pd.DataFrame(data = np.stack([np.mean(err,axis=0) for err in x_err_collect.values()],axis=1),
                                  columns = experiment_tags, index = x_indices)
        x_err_mean['metric'] = 'mean'
        x_err_std = pd.DataFrame(data = np.stack([np.std(err,axis=0) for err in x_err_collect.values()],axis=1),
                                 columns = experiment_tags, index = x_indices)
        x_err_std['metric'] = 'std'
        df_x_err = pd.concat([x_err_median,x_err_mean,x_err_std])
        
        ## summary for y
        y_indices = [f'step_{i+1}' for i in range(self.args.horizon)]
        y_err_median = pd.DataFrame(data = np.stack([np.median(err,axis=0) for err in y_err_collect.values()],axis=1),
                                    columns = experiment_tags, index = y_indices)
        y_err_median['metric'] = 'median'
        y_err_mean = pd.DataFrame(data = np.stack([np.mean(err,axis=0) for err in y_err_collect.values()],axis=1),
                                  columns = experiment_tags, index = y_indices)
        y_err_mean['metric'] = 'mean'
        y_err_std = pd.DataFrame(data = np.stack([np.std(err,axis=0) for err in y_err_collect.values()],axis=1),
                                 columns = experiment_tags, index = y_indices)
        y_err_std['metric'] = 'std'
        df_y_err = pd.concat([y_err_median,y_err_mean,y_err_std])
        
        self.err_eval = {'summary_x_err': df_x_err, 'summary_y_err': df_y_err, 'raw_x_err': x_err_collect, 'raw_y_err': y_err_collect}
        
        ## export to excel
        if export_to_excel:
            export_error_to_excel(df_x_err,df_y_err,x_err_collect,y_err_collect,self.args.output_folder)
