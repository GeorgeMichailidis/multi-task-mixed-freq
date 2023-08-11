"""
tree-based simulation trainer
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from helpers import ClsConstructor, Evaluator
from ._base_simtrainer import _baseSimTrainer

class gbmSimTrainer(_baseSimTrainer):
    
    def __init__(self, args, evaluator, seed_x = None, seed_y = None):
        
        super().__init__(args,evaluator)
        self.cls_constructor = ClsConstructor(args)
        self.seed_x = seed_x or 527
        self.seed_y = seed_y or 528
    
    def set_seed(self, repickle_args=True):
        
        self.args.hyper_params_x['seed'] = self.seed_x
        self.args.hyper_params_y['seed'] = self.seed_y
        
        if repickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    def config_and_train_model(self):
        
        model = self.cls_constructor.create_model() ## object TwoGBM()
        train_dataset, val_dataset = self.train_dataset, self.val_dataset
        
        ## train
        print(f'[{self.args.model_type} model training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        history = model.train(train_dataset, val_dataset)
        
        self.model = model
        
    def config_predictor(self):
        
        args = self.args
        model, dp = self.model, self.dp
        self.predictor = self.cls_constructor.create_predictor(model, dp, apply_inv_scaler = args.scale_data)
    
    def run_forecast(self, pickle_predictions = True):
        
        args = self.args
        predictor, dp = self.predictor, self.dp
        xdata, ydata = self.raw_data
        
        y_start_id = args.test_start_id
        x_start_id = args.freq_ratio * (y_start_id - 1)
        x_pred_recorder, y_pred_recorder = {}, {}
        
        print(f'[forecast experiments start] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        for experiment_id in range(args.test_size):
            assert x_start_id == args.freq_ratio * (y_start_id - 1)
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
                                                                
                x_pred, y_pred = predictor(inputs, x_step=x_step, horizon = args.horizon)
                
                x_pred_recorder.setdefault(experiment_tag,[]).append({'target': targets[0], 'prediction': x_pred})
                y_pred_recorder.setdefault(experiment_tag,[]).append({'target': targets[1], 'prediction': y_pred})
        
            # update x_start_id and y_start_id
            x_start_id += args.freq_ratio
            y_start_id += 1
        print(f'[forecast experiments end] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        if pickle_predictions:
            with open(f"{self.args.output_folder}/predictions.pickle","wb") as handle:
                pickle.dump({'x_target_pred': x_pred_recorder, 'y_target_pred': y_pred_recorder}, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        self.x_pred_recorder = x_pred_recorder
        self.y_pred_recorder = y_pred_recorder
    
    def end_to_end(
        self,
        repickle_args = True,
        print_train_err = True,
        plot_fitted = True,
        pickle_predictions = True,
        export_fcast_err_to_excel = True
    ):
    
        self.set_seed(repickle_args=repickle_args)
        self.source_data()
        self.generate_train_val_datasets()
        
        self.config_and_train_model()
        self.eval_training(print_train_err=print_train_err,plot_fitted=plot_fitted)
        
        self.config_predictor()
        self.run_forecast(pickle_predictions=pickle_predictions)
        
        self.eval_forecast(export_to_excel=export_fcast_err_to_excel)
