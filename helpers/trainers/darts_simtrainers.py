
import os
import pickle
import pandas as pd
import numpy as np
import pandas as pd
from datetime import datetime

from helpers import ClsConstructor, Evaluator
from ._base_simtrainer import _baseSimTrainer

import matplotlib.pyplot as plt
from itertools import chain

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import darts
from darts import TimeSeries
from darts.models import NHiTSModel, RNNModel
from darts.utils.likelihood_models import GaussianLikelihood

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

class arimaSimTrainer(_baseSimTrainer):
    
    def __init__(self, args, evaluator, seed = None):
        
        super().__init__(args,evaluator)
        self.cls_constructor = ClsConstructor(args)
        self.seed = seed or 532
        
        ## some checks for the hyperparameter setup, since here we are forecasting using a VAR system
        assert self.args.Lx == self.args.freq_ratio * self.args.Ly
        assert self.args.Tx == self.args.freq_ratio * self.args.Ty
    
    def set_seed(self, repickle_args=True):
        
        self.args.hyper_params['seed'] = self.seed
        
        if repickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    ## overrides the attribute in the base class
    def generate_train_val_datasets(self):
        
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        xcols, ycols = self.data_cols
        
        ########
        ## get training & validation data
        ## for hf, length reserved for train = freq_ratio * (train_size + 1)
        ## for lf, length reserved for train = train_size + 1
        ## The above logic is similar for val
        ########
        
        ## step 1: chop the numpy array for train/validation
        x_train_cutoff = dp.find_length(args.train_size, len_input = args.Lx, len_target = 1, stride = args.freq_ratio)
        y_train_cutoff = dp.find_length(args.train_size, len_input = args.Ly, len_target = 1, stride = 1)
    
        x_trainval_end = dp.find_length(args.train_size + args.val_size, len_input = args.Lx, len_target = args.Tx, stride = args.freq_ratio)
        y_trainval_end = dp.find_length(args.train_size + args.val_size, len_input = args.Ly, len_target = args.Ty, stride = 1)
        
        x_trainval, y_trainval = xdata[:x_trainval_end,:], ydata[:y_trainval_end,:]
        #print(f'x_trainval_end = {x_trainval_end}, y_trainval_end = {y_trainval_end}. NOTE: ending idx is exclusive')
        
        x_train_ds, y_train_ds, x_val_ds, y_val_ds = [], [], [], []
        ## step 2: create dataset
        for ix in range(self.args.dim_x):
            train, val = self._make_trainval_dataset(x_trainval[:,ix], x_train_cutoff)
            x_train_ds.append(train)
            x_val_ds.append(val)
        
        for iy in range(self.args.dim_y):
            train, val = self._make_trainval_dataset(y_trainval[:,iy], y_train_cutoff)
            y_train_ds.append(train)
            y_val_ds.append(val)
            
        self.dp = dp
        self.train_datasets = (x_train_ds, y_train_ds)
        self.val_datasets = (x_val_ds, y_val_ds)
        return
        
    def _make_trainval_dataset(self, values, training_cutoff):
        """
        sub-routine for generate_train_val_datasets()
        returns the dataset object TimeSeriesDataSet) for the system
        """
        dataset = TimeSeries.from_values(values).astype(np.float32)
        train_dataset, val_dataset = dataset.split_after(training_cutoff)
        
        return train_dataset, val_dataset
    
    def _create_trainers(self):
    
        ## element-wise
        x_trainers = [darts.models.forecasting.arima.ARIMA(p=self.args.hyper_params_x['p'],
                                                d=self.args.hyper_params_x['d'],
                                                q=self.args.hyper_params_x['q']) for ix in range(self.args.dim_x)]
                      
        y_trainers = [darts.models.forecasting.arima.ARIMA(p=self.args.hyper_params_y['p'],
                                                 d=self.args.hyper_params_y['d'],
                                                 q=self.args.hyper_params_y['q']) for iy in range(self.args.dim_y)]
        
        return x_trainers, y_trainers
        
    def config_and_train_model(self):
    
        ## train each network
        x_trainers, y_trainers = self._create_trainers()
        
        print(f'[x_arimas training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        for ix in range(self.args.dim_x):
            x_trainers[ix].fit(self.train_datasets[0][ix])
        
        print(f'[y_arimas training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        for iy in range(self.args.dim_y):
            y_trainers[iy].fit(self.train_datasets[1][iy])
        
        self.models = (x_trainers, y_trainers)
        return
        
    def eval_training(self):
        
        x_col, y_col = np.random.choice(list(range(self.args.dim_x))), np.random.choice(list(range(self.args.dim_y)))
        
        ##########
        ## fit for the training set
        ##########
        print(f'[evaluation on the training set] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        x_train_prediction = self.models[0][x_col].historical_forecasts(series=self.train_datasets[0][x_col], forecast_horizon=self.args.Tx, stride=5, retrain=False, verbose=False)
        y_train_prediction = self.models[1][y_col].historical_forecasts(series=self.train_datasets[1][y_col], forecast_horizon=self.args.Ty, stride=1, retrain=False, verbose=False)

        fig_train, axs = plt.subplots(1, 2, figsize=(12,3), constrained_layout=True)
        
        ## plot for y
        self.train_datasets[1][y_col].pd_series().plot(ax=axs[0],label='truth')
        y_train_prediction.pd_series().plot(ax=axs[0],label='pred')
        axs[0].set_title(f'y_col={y_col}')
        axs[0].legend()
        
        ## plot for x
        self.train_datasets[0][x_col].pd_series().plot(ax=axs[1],label='truth')
        x_train_prediction.pd_series().plot(ax=axs[0],label='pred')
        axs[1].set_title(f'x_col={x_col}')
        axs[1].legend()
            
        fig_train.savefig(f'{self.args.output_folder}/train_fit_rand_col.png')
        plt.close()
        
        ###########
        ## fit for the validation set
        ###########
        print(f'[evaluation on the validation set] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        x_val_prediction = self.models[0][x_col].historical_forecasts(series=self.val_datasets[0][x_col], forecast_horizon=self.args.Tx, stride=5, retrain=False, verbose=False)
        y_val_prediction = self.models[1][y_col].historical_forecasts(series=self.val_datasets[1][y_col], forecast_horizon=self.args.Ty, stride=1, retrain=False, verbose=False)

        fig_val, axs = plt.subplots(1, 2, figsize=(12,3), constrained_layout=True)
        
        ## plot for y
        self.val_datasets[1][y_col].pd_series().plot(ax=axs[0],label='truth')
        y_val_prediction.pd_series().plot(ax=axs[0],label='pred')
        axs[0].set_title(f'y_col={y_col}')
        axs[0].legend()
        
        ## plot for x
        self.val_datasets[0][x_col].pd_series().plot(ax=axs[1],label='truth')
        x_val_prediction.pd_series().plot(ax=axs[0],label='pred')
        axs[1].set_title(f'x_col={x_col}')
        axs[1].legend()
            
        fig_val.savefig(f'{self.args.output_folder}/val_fit_rand_col.png')
        plt.close()
        
        return
        
    def run_forecast(self, pickle_predictions = True):
        
        args = self.args
        xdata, ydata = self.raw_data
        
        y_start_id = args.test_start_id
        x_start_id = args.freq_ratio * (args.test_start_id - 1)
        
        x_pred_recorder, y_pred_recorder = {}, {}
        
        print(f'[forecast experiments start] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        for experiment_id in range(args.test_size):
            
            if args.verbose > 0:
                if not (experiment_id+1)%args.verbose:
                    print(f'running {experiment_id+1}-th experiment out of {args.test_size}')
        
            data_lf = ydata[y_start_id:(y_start_id + args.Ly + args.Ty)]
            y_target = data_lf[-args.Ty:,:]
            
            y_pred = []
            for iy in range(args.dim_y):
                y_test_dataset = TimeSeries.from_values(data_lf[:args.Ly,iy]).astype(np.float32)
                y_pred_iy = self.models[1][iy].predict(n=args.Ty, series=y_test_dataset, verbose=False).values()
                y_pred.append(y_pred_iy)
            y_pred = np.concatenate(y_pred,axis=1)
            
            for x_step in range(args.freq_ratio+1):
                experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
                
                data_hf = xdata[(x_start_id + x_step):(x_start_id + args.Lx + args.Tx)]
                x_target = data_hf[-args.Tx:,:]
                
                x_pred = []
                for ix in range(args.dim_x):
                    x_test_dataset = TimeSeries.from_values(data_hf[:args.Lx,ix]).astype(np.float32)
                    x_pred_ix = self.models[0][ix].predict(n=args.Tx-x_step, series=x_test_dataset, verbose=False).values()
                    x_pred.append(x_pred_ix)
                x_pred = np.concatenate(x_pred,axis=1)
                
                x_pred = np.concatenate([x_target[:x_step], x_pred], axis=0)
                ## record
                y_pred_recorder.setdefault(experiment_tag,[]).append({'target': y_target, 'prediction': y_pred})
                x_pred_recorder.setdefault(experiment_tag,[]).append({'target': x_target, 'prediction': x_pred})
            
            # update x_start_id and y_start_id
            x_start_id += args.freq_ratio
            y_start_id += 1
        
        if pickle_predictions:
            with open(f"{self.args.output_folder}/predictions.pickle","wb") as handle:
                pickle.dump({'x_target_pred': x_pred_recorder, 'y_target_pred': y_pred_recorder}, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        self.x_pred_recorder = x_pred_recorder
        self.y_pred_recorder = y_pred_recorder
        
        return

class deepARSimTrainer(_baseSimTrainer):
    
    def __init__(self, args, evaluator, seed = None):
        
        super().__init__(args,evaluator)
        self.cls_constructor = ClsConstructor(args)
        self.cuda = getattr(args, 'cuda', 0)
        self.seed = seed or 532
        
        if torch.cuda.is_available():
            self.accelerator = 'gpu'
        #elif torch.has_mps:
        #    self.accelerator = 'mps'
        else:
            self.accelerator = 'cpu'
        self.devices = [self.cuda] if self.accelerator == 'gpu' else 1
    
        ## some checks for the hyperparameter setup, since here we are forecasting using a VAR system
        assert self.args.Lx == self.args.freq_ratio * self.args.Ly
        assert self.args.Tx == self.args.freq_ratio * self.args.Ty
    
    def set_seed(self, repickle_args=True):
        
        self.args.hyper_params['seed'] = self.seed
        
        if repickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    ## overrides the attribute in the base class
    def generate_train_val_datasets(self):
        
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        xcols, ycols = self.data_cols
        
        ########
        ## get training & validation data
        ## for hf, length reserved for train = freq_ratio * (train_size + 1)
        ## for lf, length reserved for train = train_size + 1
        ## The above logic is similar for val
        ########
        
        ## step 1: chop the numpy array for train/validation
        train_cutoff = dp.find_length(args.train_size, len_input = args.Ly, len_target = args.Ty, stride = 1)
        
        x_trainval_end = dp.find_length(args.train_size + args.val_size, len_input = args.Lx, len_target = args.Tx, stride = args.freq_ratio)
        y_trainval_end = dp.find_length(args.train_size + args.val_size, len_input = args.Ly, len_target = args.Ty, stride = 1)
        
        x_trainval, y_trainval = xdata[:x_trainval_end,:], ydata[:y_trainval_end,:]
        #print(f'x_trainval_end = {x_trainval_end}, y_trainval_end = {y_trainval_end}. NOTE: ending idx is exclusive')
        
        dfs, train_datasets, val_datasets = [], [], []
        ## step 2: prepare datasets for different vintages
        for x_step in range(args.freq_ratio + 1):
            ## obtain data frame of the designamted format
            system_df = self._prepare_system_df(data_hf = x_trainval, data_lf = y_trainval, x_step = x_step)
            ## create dataset
            train_ds, val_ds = self._make_trainval_dataset(system_df, train_cutoff)
            
            dfs.append(system_df)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            
        self.dp = dp
        self.train_datasets = tuple(train_datasets)
        self.val_datasets = tuple(val_datasets)
        
        return dfs
    
    def _freq_alignment(self, data_hf):
        """
        do frequency alignment for high frequency
        """
        df_hf = pd.DataFrame(data=data_hf,columns=self.data_cols[0])
        df_hf.insert(0, 'period_idx',[i//self.args.freq_ratio for i in range(df_hf.shape[0])])
        df_hf.insert(1, 'vintage', df_hf.groupby(['period_idx']).cumcount())
        
        ## instead of using pivot_table where the columns will be sorted
        df_hf_aligned = []
        for vintage_id in range(self.args.freq_ratio):
            df_partial = df_hf[df_hf['vintage'] == vintage_id].copy().drop(columns=['vintage']).set_index("period_idx")
            df_partial.columns = [(col, vintage_id) for col in df_partial.columns]
            df_hf_aligned.append(df_partial)
            
        df_hf_aligned = pd.concat(df_hf_aligned,axis=1)
        return df_hf_aligned
    
    def _shift(self, df_aligned, x_step=0, reverse=False):
        direction = -1 if not reverse else 1
        
        if x_step > 0:
            df_aligned_shifted = {}
            for col in df_aligned.columns:
                if col[1] < x_step:
                    df_aligned_shifted[col] = df_aligned[col].shift(direction)
                else:
                    df_aligned_shifted[col] = df_aligned[col]
            
            df_aligned = pd.concat(df_aligned_shifted,axis=1)
            df_aligned.columns = [(col,vintage_id) for col, vintage_id in df_aligned.columns]
        
        return df_aligned
        
    def _prepare_system_df(self, data_hf, data_lf, x_step = 0):
        """
        sub-routine for generate_train_val_datasets()
        prepare data to the format that is suitable for the data loader
        * for the high-frequency variables, they are modelled as an autoregressive system;
        * for the low-frequency variables, they are modelled as a VAR-X system
        
        argvs:
        - data_hf: high frequency numpy array of dimension (freq_ratio * (T+1), p)
        - data_lf: low frequency numpy array of dimension ((T+1), q)
        """
        
        ##########
        ## step 1: do frequency alignment for high frequency
        ##########
        df_hf_aligned = self._freq_alignment(data_hf)
        ## shift based on whether it's forecast/nowcasts
        df_hf_aligned_shifted = self._shift(df_hf_aligned, x_step=x_step, reverse=False)
        ## rename; otherwise multilevel col names will be problematic for melting
        df_hf_aligned_shifted.columns = [f'{col}_{vintage_id}' for col, vintage_id in df_hf_aligned_shifted.columns]
        
        ##########
        ## step 2: prepare df for the system
        ##########
        df_lf = pd.DataFrame(data=data_lf,columns=self.data_cols[1],index=list(range(data_lf.shape[0])))
        df_lf.index.name = 'time_idx'
        
        df_system = df_lf.join(df_hf_aligned_shifted, how='left').dropna()
        
        return df_system
        
    def _make_trainval_dataset(self, df, training_cutoff):
        """
        sub-routine for generate_train_val_datasets()
        returns the dataset object TimeSeriesDataSet) for the system
        """
        dataset = TimeSeries.from_values(df.values).astype(np.float32)
        train_dataset, val_dataset = dataset.split_after(training_cutoff)
        
        return train_dataset, val_dataset
    
    def _create_trainer(self):
    
        """ helper function for creating a trainer """
        early_stopper = EarlyStopping(monitor = "val_loss",
                                    min_delta = 1e-4,
                                    patience = self.args.ES_patience,
                                    verbose = True,
                                    mode = "min")
        
        trainer = RNNModel(model="LSTM",
                    hidden_dim=self.args.hyper_params['hidden_dim'],
                    n_rnn_layers=self.args.hyper_params['n_rnn_layers'],
                    dropout=self.args.hyper_params['dropout_rate'],
                    batch_size=self.args.batch_size,
                    n_epochs=self.args.epochs,
                    random_state=0,
                    training_length=self.args.Ly,
                    input_chunk_length=self.args.Ly,
                    likelihood=GaussianLikelihood(),
                    optimizer_kwargs={"lr": self.args.learning_rate},
                    lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    lr_scheduler_kwargs={"optimizer": torch.optim.Adam,
                                         "mode": "min",
                                         "factor": self.args.hyper_params['reduce_on_plateau_factor'],
                                         "threshold": 0.0001,
                                         "verbose": True,
                                         "patience": self.args.hyper_params['reduce_on_plateau_patience']},
                    pl_trainer_kwargs={"callbacks": [early_stopper],
                                       "accelerator": self.accelerator,
                                       "devices": self.devices,
                                       "gradient_clip_val": self.args.gradient_clip})
                    
        return trainer
        
    def config_and_train_model(self):
    
        ## train each network
        trainers = []
        for vintage_id in range(self.args.freq_ratio + 1):
        
            trainer = self._create_trainer()
            
            print(f'[network_{vintage_id} training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            trainer.fit(self.train_datasets[vintage_id],val_series=self.val_datasets[vintage_id],verbose=False)
            print(f'>> trainer stopped')
                       
            trainers.append(trainer)
        
        ## save down train and validation data loader
        self.models = trainers
        return trainers
        
    def eval_training(self):
        
        x_col, y_col = np.random.choice(list(range(self.args.dim_x))), np.random.choice(list(range(self.args.dim_y)))
        vintage_id = 0 ## here we only evaluate forecast performance, hence set vintage_id = 0
        model_to_use = self.models[vintage_id]
        
        ##########
        ## fit for the training set
        ##########
        print(f'[training evaluation on the training set] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        raw_prediction = model_to_use.historical_forecasts(series=self.train_datasets[vintage_id], forecast_horizon=self.args.Ty, stride=5, retrain=False, verbose=False)

        fig_train, axs = plt.subplots(2, 2, figsize=(12,6), constrained_layout=True)
        
        ## plot for y
        self.train_datasets[vintage_id].univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='truth')
        raw_prediction.univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='pred')
        axs[0,0].set_title(f'y_col={y_col}')
        axs[0,0].legend()
        
        ## plot for different aligned coordinates of x
        for seq_id in range(self.args.freq_ratio):
            ax_id = (seq_id + 1)//2, (seq_id + 1)%2
            x_coord = y_col + seq_id*self.args.dim_x+x_col
            self.train_datasets[vintage_id].univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='truth')
            raw_prediction.univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='pred')
            axs[ax_id].set_title(f'x_col={x_col},seq_id={seq_id}')
            axs[ax_id].legend()
            
        fig_train.savefig(f'{self.args.output_folder}/train_fit_rand_col.png')
        plt.close()
        
        ###########
        ## fit for the validation set
        ###########
        print(f'[training evaluation on the validation set] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        raw_prediction = model_to_use.historical_forecasts(series=self.val_datasets[vintage_id], forecast_horizon=self.args.Ty, stride=5, retrain=False, verbose=False)

        fig_val, axs = plt.subplots(2, 2, figsize=(12,6), constrained_layout=True)
        ## plot for y
        self.val_datasets[vintage_id].univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='truth')
        raw_prediction.univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='pred')
        axs[0,0].set_title(f'y_col={y_col}')
        axs[0,0].legend()
        
        ## plot for different aligned coordinates of x
        for seq_id in range(self.args.freq_ratio):
            ax_id = (seq_id + 1)//2, (seq_id + 1)%2
            x_coord = y_col + seq_id*self.args.dim_x+x_col
            self.val_datasets[vintage_id].univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='truth')
            raw_prediction.univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='pred')
            axs[ax_id].set_title(f'x_col={x_col},seq_id={seq_id}')
            axs[ax_id].legend()
            
        fig_val.savefig(f'{self.args.output_folder}/val_fit_rand_col.png')
        plt.close()
        
    def run_forecast(self, pickle_predictions = True):
        
        args = self.args
        xdata, ydata = self.raw_data
        
        y_start_id = args.test_start_id
        x_start_id = args.freq_ratio * (args.test_start_id - 1)
        
        x_pred_recorder, y_pred_recorder = {}, {}
        
        print(f'[forecast experiments start] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        for experiment_id in range(args.test_size):
        
            for x_step in range(args.freq_ratio+1):
                experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
                
                data_hf = xdata[x_start_id:(x_start_id + args.Lx + args.Tx)]
                x_target = data_hf[-args.Tx:,:]
            
                data_lf = ydata[y_start_id:(y_start_id + args.Ly + args.Ty)]
                y_target = data_lf[-args.Ty:,:]
                
                df_system = self._prepare_system_df(data_hf, data_lf, x_step = x_step)
                test_dataset = TimeSeries.from_values(df_system.values[:args.Ly,:]).astype(np.float32)
                
                model_to_use = self.models[x_step]
                raw_pred = model_to_use.predict(n=args.Ty, series=test_dataset, num_samples=100, verbose=False)
                yx_pred = raw_pred.quantile_timeseries(0.50).values()
                
                ## y comes first, so get that
                y_pred = yx_pred[:,:self.args.dim_y]
                y_pred_recorder.setdefault(experiment_tag,[]).append({'target': y_target, 'prediction': y_pred})
                
                ## for x, need to undo the shift and reverse freq_alignment
                x_pred_raw = yx_pred[:,self.args.dim_y:]
                x_pred = self._extract_x_predictions(x_pred_raw, x_step=x_step)
                x_pred[:x_step] = x_target[:x_step]
                
                x_pred_recorder.setdefault(experiment_tag,[]).append({'target': x_target, 'prediction': x_pred})
            
            # update x_start_id and y_start_id
            x_start_id += args.freq_ratio
            y_start_id += 1
        
        if pickle_predictions:
            with open(f"{self.args.output_folder}/predictions.pickle","wb") as handle:
                pickle.dump({'x_target_pred': x_pred_recorder, 'y_target_pred': y_pred_recorder}, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        self.x_pred_recorder = x_pred_recorder
        self.y_pred_recorder = y_pred_recorder

    def _extract_x_predictions(self, x_pred_raw, x_step=0):
    
        assert x_pred_raw.shape == (self.args.Ty, self.args.freq_ratio * self.args.dim_x)
        cols_raw = list(chain.from_iterable([ [(col,vintage_id) for col in self.data_cols[0]] for vintage_id in range(self.args.freq_ratio)]))
        
        df_pred_aligned = pd.DataFrame(data = x_pred_raw, columns = cols_raw, index = list(range(x_pred_raw.shape[0])))
        df_pred_aligned.index.name = 'period_idx'
        ## shift x_step
        df_pred_aligned = self._shift(df_pred_aligned, x_step=x_step, reverse=True)
        
        ## undo freq_alignment
        df_pred_melted = pd.melt(df_pred_aligned.reset_index(), id_vars='period_idx')
        df_pred_melted['time_idx'] = df_pred_melted.apply(lambda x: x['variable'][1] + x['period_idx'] * self.args.freq_ratio, axis=1)
        df_pred_melted['variable'] = df_pred_melted['variable'].apply(lambda x: x[0])
        
        df_pred = []
        for variable in self.data_cols[0]:
            df_this_variable = df_pred_melted.loc[df_pred_melted['variable']==variable].copy().sort_values(by=['time_idx']).set_index('time_idx')
            df_pred.append(df_this_variable.drop(columns=['period_idx','variable']).rename(columns={'value':variable}))
        
        df_pred = pd.concat(df_pred,axis=1)
        
        return df_pred.values

class nhitsSimTrainer(_baseSimTrainer):
    
    def __init__(self, args, evaluator, seed = None):
        
        super().__init__(args,evaluator)
        self.cls_constructor = ClsConstructor(args)
        self.cuda = getattr(args, 'cuda', 0)
        self.seed = seed or 532
        
        if torch.cuda.is_available():
            self.accelerator = 'gpu'
        #elif torch.has_mps:
        #    self.accelerator = 'mps'
        else:
            self.accelerator = 'cpu'
        self.devices = [self.cuda] if self.accelerator == 'gpu' else 1
    
        ## some checks for the hyperparameter setup, since here we are forecasting using a VAR system
        assert self.args.Lx == self.args.freq_ratio * self.args.Ly
        assert self.args.Tx == self.args.freq_ratio * self.args.Ty
    
    def set_seed(self, repickle_args=True):
        
        self.args.hyper_params['seed'] = self.seed
        
        if repickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    ## overrides the attribute in the base class
    def generate_train_val_datasets(self):
        
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        xcols, ycols = self.data_cols
        
        ########
        ## get training & validation data
        ## for hf, length reserved for train = freq_ratio * (train_size + 1)
        ## for lf, length reserved for train = train_size + 1
        ## The above logic is similar for val
        ########
        
        ## step 1: chop the numpy array for train/validation
        train_cutoff = dp.find_length(args.train_size, len_input = args.Ly, len_target = args.Ty, stride = 1)
        
        x_trainval_end = dp.find_length(args.train_size + args.val_size, len_input = args.Lx, len_target = args.Tx, stride = args.freq_ratio)
        y_trainval_end = dp.find_length(args.train_size + args.val_size, len_input = args.Ly, len_target = args.Ty, stride = 1)
        
        x_trainval, y_trainval = xdata[:x_trainval_end,:], ydata[:y_trainval_end,:]
        #print(f'x_trainval_end = {x_trainval_end}, y_trainval_end = {y_trainval_end}. NOTE: ending idx is exclusive')
        
        dfs, train_datasets, val_datasets = [], [], []
        ## step 2: prepare datasets for different vintages
        for x_step in range(args.freq_ratio + 1):
            ## obtain data frame of the designamted format
            system_df = self._prepare_system_df(data_hf = x_trainval, data_lf = y_trainval, x_step = x_step)
            ## create dataset
            train_ds, val_ds = self._make_trainval_dataset(system_df, train_cutoff)
            
            dfs.append(system_df)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            
        self.dp = dp
        self.train_datasets = tuple(train_datasets)
        self.val_datasets = tuple(val_datasets)
        
        return dfs
    
    def _freq_alignment(self, data_hf):
        """
        do frequency alignment for high frequency
        """
        df_hf = pd.DataFrame(data=data_hf,columns=self.data_cols[0])
        df_hf.insert(0, 'period_idx',[i//self.args.freq_ratio for i in range(df_hf.shape[0])])
        df_hf.insert(1, 'vintage', df_hf.groupby(['period_idx']).cumcount())
        
        ## instead of using pivot_table where the columns will be sorted
        df_hf_aligned = []
        for vintage_id in range(self.args.freq_ratio):
            df_partial = df_hf[df_hf['vintage'] == vintage_id].copy().drop(columns=['vintage']).set_index("period_idx")
            df_partial.columns = [(col, vintage_id) for col in df_partial.columns]
            df_hf_aligned.append(df_partial)
            
        df_hf_aligned = pd.concat(df_hf_aligned,axis=1)
        return df_hf_aligned
    
    def _shift(self, df_aligned, x_step=0, reverse=False):
        direction = -1 if not reverse else 1
        
        if x_step > 0:
            df_aligned_shifted = {}
            for col in df_aligned.columns:
                if col[1] < x_step:
                    df_aligned_shifted[col] = df_aligned[col].shift(direction)
                else:
                    df_aligned_shifted[col] = df_aligned[col]
            
            df_aligned = pd.concat(df_aligned_shifted,axis=1)
            df_aligned.columns = [(col,vintage_id) for col, vintage_id in df_aligned.columns]
        
        return df_aligned
        
    def _prepare_system_df(self, data_hf, data_lf, x_step = 0):
        """
        sub-routine for generate_train_val_datasets()
        prepare data to the format that is suitable for the data loader
        * for the high-frequency variables, they are modelled as an autoregressive system;
        * for the low-frequency variables, they are modelled as a VAR-X system
        
        argvs:
        - data_hf: high frequency numpy array of dimension (freq_ratio * (T+1), p)
        - data_lf: low frequency numpy array of dimension ((T+1), q)
        """
        
        ##########
        ## step 1: do frequency alignment for high frequency
        ##########
        df_hf_aligned = self._freq_alignment(data_hf)
        ## shift based on whether it's forecast/nowcasts
        df_hf_aligned_shifted = self._shift(df_hf_aligned, x_step=x_step, reverse=False)
        ## rename; otherwise multilevel col names will be problematic for melting
        df_hf_aligned_shifted.columns = [f'{col}_{vintage_id}' for col, vintage_id in df_hf_aligned_shifted.columns]
        
        ##########
        ## step 2: prepare df for the system
        ##########
        df_lf = pd.DataFrame(data=data_lf,columns=self.data_cols[1],index=list(range(data_lf.shape[0])))
        df_lf.index.name = 'time_idx'
        
        df_system = df_lf.join(df_hf_aligned_shifted, how='left').dropna()
        
        return df_system
        
    def _make_trainval_dataset(self, df, training_cutoff):
        """
        sub-routine for generate_train_val_datasets()
        returns the dataset object TimeSeriesDataSet) for the system
        """
        dataset = TimeSeries.from_values(df.values).astype(np.float32)
        train_dataset, val_dataset = dataset.split_after(training_cutoff)
        
        return train_dataset, val_dataset
    
    def _create_trainer(self):
    
        """ helper function for creating a trainer """
        early_stopper = EarlyStopping(monitor = "val_loss",
                                    min_delta = 1e-4,
                                    patience = self.args.ES_patience,
                                    verbose = True,
                                    mode = "min")
        
        trainer = NHiTSModel(input_chunk_length = self.args.Ly,
                      output_chunk_length = self.args.Ty,
                      num_stacks=self.args.hyper_params['num_stacks'],
                      num_blocks=self.args.hyper_params['num_blocks'],
                      num_layers=self.args.hyper_params['num_layers'],
                      layer_widths=self.args.hyper_params['layer_widths'],
                      pooling_kernel_sizes=None,
                      n_freq_downsample=None,
                      dropout=self.args.hyper_params['dropout_rate'],
                      activation='ReLU',
                      MaxPool1d=True,
                      batch_size=self.args.batch_size,
                      n_epochs=self.args.epochs,
                      optimizer_kwargs={"lr": self.args.learning_rate},
                      lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                      lr_scheduler_kwargs={"optimizer": torch.optim.Adam,
                                           "mode": "min",
                                           "factor": self.args.hyper_params['reduce_on_plateau_factor'],
                                           "threshold": 0.0001,
                                           "verbose": True,
                                           "patience": self.args.hyper_params['reduce_on_plateau_patience']},
                      pl_trainer_kwargs={"callbacks": [early_stopper],
                                         "accelerator": self.accelerator,
                                         "devices": self.devices,
                                         "gradient_clip_val": self.args.gradient_clip}
                      )
        
        return trainer
        
    def config_and_train_model(self):
    
        ## train each network
        trainers = []
        for vintage_id in range(self.args.freq_ratio + 1):
        
            trainer = self._create_trainer()
            
            print(f'[network_{vintage_id} training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            trainer.fit(self.train_datasets[vintage_id],val_series=self.val_datasets[vintage_id],verbose=False)
            print(f'>> trainer stopped')
                       
            trainers.append(trainer)
        
        ## save down train and validation data loader
        self.models = trainers
        return trainers
        
    def eval_training(self):
        
        x_col, y_col = np.random.choice(list(range(self.args.dim_x))), np.random.choice(list(range(self.args.dim_y)))
        vintage_id = 0 ## here we only evaluate forecast performance, hence set vintage_id = 0
        model_to_use = self.models[vintage_id]
        
        ##########
        ## fit for the training set
        ##########
        print(f'[training evaluation on the training set] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        raw_prediction = model_to_use.historical_forecasts(series=self.train_datasets[vintage_id], forecast_horizon=self.args.Ty, stride=5, retrain=False, verbose=False)

        fig_train, axs = plt.subplots(2, 2, figsize=(12,6), constrained_layout=True)
        
        ## plot for y
        self.train_datasets[vintage_id].univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='truth')
        raw_prediction.univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='pred')
        axs[0,0].set_title(f'y_col={y_col}')
        axs[0,0].legend()
        
        ## plot for different aligned coordinates of x
        for seq_id in range(self.args.freq_ratio):
            ax_id = (seq_id + 1)//2, (seq_id + 1)%2
            x_coord = y_col + seq_id*self.args.dim_x+x_col
            self.train_datasets[vintage_id].univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='truth')
            raw_prediction.univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='pred')
            axs[ax_id].set_title(f'x_col={x_col},seq_id={seq_id}')
            axs[ax_id].legend()
            
        fig_train.savefig(f'{self.args.output_folder}/train_fit_rand_col.png')
        plt.close()
        
        ###########
        ## fit for the validation set
        ###########
        print(f'[training evaluation on the validation set] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        raw_prediction = model_to_use.historical_forecasts(series=self.val_datasets[vintage_id], forecast_horizon=self.args.Ty, stride=5, retrain=False, verbose=False)

        fig_val, axs = plt.subplots(2, 2, figsize=(12,6), constrained_layout=True)
        ## plot for y
        self.val_datasets[vintage_id].univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='truth')
        raw_prediction.univariate_component(y_col).pd_series().plot(ax=axs[0,0],label='pred')
        axs[0,0].set_title(f'y_col={y_col}')
        axs[0,0].legend()
        
        ## plot for different aligned coordinates of x
        for seq_id in range(self.args.freq_ratio):
            ax_id = (seq_id + 1)//2, (seq_id + 1)%2
            x_coord = y_col + seq_id*self.args.dim_x+x_col
            self.val_datasets[vintage_id].univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='truth')
            raw_prediction.univariate_component(x_coord).pd_series().plot(ax=axs[ax_id],label='pred')
            axs[ax_id].set_title(f'x_col={x_col},seq_id={seq_id}')
            axs[ax_id].legend()
            
        fig_val.savefig(f'{self.args.output_folder}/val_fit_rand_col.png')
        plt.close()
        
    def run_forecast(self, pickle_predictions = True):
        
        args = self.args
        xdata, ydata = self.raw_data
        
        y_start_id = args.test_start_id
        x_start_id = args.freq_ratio * (args.test_start_id - 1)
        
        x_pred_recorder, y_pred_recorder = {}, {}
        
        print(f'[forecast experiments start] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        for experiment_id in range(args.test_size):
        
            for x_step in range(args.freq_ratio+1):
                experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
                
                data_hf = xdata[x_start_id:(x_start_id + args.Lx + args.Tx)]
                x_target = data_hf[-args.Tx:,:]
            
                data_lf = ydata[y_start_id:(y_start_id + args.Ly + args.Ty)]
                y_target = data_lf[-args.Ty:,:]
                
                df_system = self._prepare_system_df(data_hf, data_lf, x_step = x_step)
                test_dataset = TimeSeries.from_values(df_system.values[:args.Ly,:]).astype(np.float32)
                
                model_to_use = self.models[x_step]
                yx_pred = model_to_use.predict(n=args.Ty, series=test_dataset, verbose=False).values()
                
                ## y comes first, so get that
                y_pred = yx_pred[:,:self.args.dim_y]
                y_pred_recorder.setdefault(experiment_tag,[]).append({'target': y_target, 'prediction': y_pred})
                
                ## for x, need to undo the shift and reverse freq_alignment
                x_pred_raw = yx_pred[:,self.args.dim_y:]
                x_pred = self._extract_x_predictions(x_pred_raw, x_step=x_step)
                x_pred[:x_step] = x_target[:x_step]
                
                x_pred_recorder.setdefault(experiment_tag,[]).append({'target': x_target, 'prediction': x_pred})
            
            # update x_start_id and y_start_id
            x_start_id += args.freq_ratio
            y_start_id += 1
        
        if pickle_predictions:
            with open(f"{self.args.output_folder}/predictions.pickle","wb") as handle:
                pickle.dump({'x_target_pred': x_pred_recorder, 'y_target_pred': y_pred_recorder}, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        self.x_pred_recorder = x_pred_recorder
        self.y_pred_recorder = y_pred_recorder

    def _extract_x_predictions(self, x_pred_raw, x_step=0):
    
        assert x_pred_raw.shape == (self.args.Ty, self.args.freq_ratio * self.args.dim_x)
        cols_raw = list(chain.from_iterable([ [(col,vintage_id) for col in self.data_cols[0]] for vintage_id in range(self.args.freq_ratio)]))
        
        df_pred_aligned = pd.DataFrame(data = x_pred_raw, columns = cols_raw, index = list(range(x_pred_raw.shape[0])))
        df_pred_aligned.index.name = 'period_idx'
        ## shift x_step
        df_pred_aligned = self._shift(df_pred_aligned, x_step=x_step, reverse=True)
        
        ## undo freq_alignment
        df_pred_melted = pd.melt(df_pred_aligned.reset_index(), id_vars='period_idx')
        df_pred_melted['time_idx'] = df_pred_melted.apply(lambda x: x['variable'][1] + x['period_idx'] * self.args.freq_ratio, axis=1)
        df_pred_melted['variable'] = df_pred_melted['variable'].apply(lambda x: x[0])
        
        df_pred = []
        for variable in self.data_cols[0]:
            df_this_variable = df_pred_melted.loc[df_pred_melted['variable']==variable].copy().sort_values(by=['time_idx']).set_index('time_idx')
            df_pred.append(df_this_variable.drop(columns=['period_idx','variable']).rename(columns={'value':variable}))
        
        df_pred = pd.concat(df_pred,axis=1)
        
        return df_pred.values
