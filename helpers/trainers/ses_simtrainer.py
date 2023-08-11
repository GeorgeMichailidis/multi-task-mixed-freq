"""
Trainer for using simple exponential smoother on synthetic data
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from ._base_simtrainer import _baseSimTrainer
from models import SimpleExpSmoother

class sesSimTrainer(_baseSimTrainer):

    def __init__(self,args,evaluator):
        super().__init__(args,evaluator)
        setattr(self, 'generate_train_val_datasets', None)
        setattr(self, 'eval_training', None)
        
    def source_data(self):
        x = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='x')
        y = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='y')
        xdata, ydata = x.values, y.values
        self.raw_data = (xdata, ydata)
    
    def run_forecast(self, pickle_predictions=True):
        
        args = self.args
        ses = SimpleExpSmoother(alpha=args.alpha)
        
        xdata, ydata = self.raw_data

        y_start_id = args.test_start_id
        x_start_id = args.freq_ratio * (y_start_id - 1)
        x_pred_recorder, y_pred_recorder = {}, {}
        
        print(f'[forecast experiments start] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        for experiment_id in range(args.test_size):
            assert x_start_id == args.freq_ratio * (y_start_id - 1)
            
            y_pred = ses.forecast(ydata[(y_start_id - args.train_size):y_start_id,:], args.horizon)
            for x_step in range(args.freq_ratio+1):
                experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
                
                x_input_indices = range(x_start_id-args.train_size,x_start_id+x_step)
                x_forecast_raw = ses.forecast(xdata[x_input_indices,:], args.freq_ratio * args.horizon - x_step)
                x_pred = np.concatenate([xdata[x_start_id: x_start_id+x_step,:], x_forecast_raw],axis=0)
                
                x_pred_recorder.setdefault(experiment_tag,[]).append({'target': xdata[x_start_id:(x_start_id+args.freq_ratio * args.horizon)], 'prediction': x_pred})
                y_pred_recorder.setdefault(experiment_tag,[]).append({'target': ydata[y_start_id:(y_start_id + args.horizon)], 'prediction': y_pred})
        
            # update x_start_id and y_start_id
            x_start_id += args.freq_ratio
            y_start_id += 1
        
        print(f'[forecast experiments end] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.x_pred_recorder = x_pred_recorder
        self.y_pred_recorder = y_pred_recorder

        if pickle_predictions:
            with open(f"{self.args.output_folder}/predictions.pickle","wb") as handle:
                pickle.dump({'x_target_pred': x_pred_recorder, 'y_target_pred': y_pred_recorder}, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def end_to_end(self, pickle_predictions = True, export_fcast_err_to_excel = True):
    
        self.source_data()
        self.run_forecast(pickle_predictions=pickle_predictions)
        self.eval_forecast(export_to_excel=export_fcast_err_to_excel)
