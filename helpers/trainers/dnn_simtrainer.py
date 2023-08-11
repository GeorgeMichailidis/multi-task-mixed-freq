"""
Trainer for tree and NN-based estimation on synthetic data
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from helpers import ClsConstructor, Evaluator
from ._base_simtrainer import _baseSimTrainer
from .utils_train import *

class LossPrintCallback(tf.keras.callbacks.Callback):
    def __init__(self,every_n_epochs=100):
        super(LossPrintCallback,self).__init__()
        self.every_n_epochs = every_n_epochs
    def on_epoch_end(self,epoch,logs=None):
        if logs.get('output_1_loss') is not None and logs.get('output_2_loss') is not None:
            if (epoch+1)%self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}; output_1_loss = {logs.get('output_1_loss'):.4f}, output_2_loss = {logs.get('output_2_loss'):.4f}.")
        else:
            if (epoch+1)%self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}.")

class nnSimTrainer(_baseSimTrainer):

    def __init__(
        self,
        args,
        evaluator=Evaluator(),
        criterion = tf.keras.losses.MeanSquaredError(),
        seed = None
    ):
        
        super().__init__(args,evaluator)
        self.criterion = criterion
        self.cls_constructor = ClsConstructor(args)
        
        if seed is None:
            default_seeds = {'MTMFSeq2Seq': 522, 'transformer': 523, 'MTMFSeq2One': 524, 'MLP': 525, 'RNN': 526}
            seed = default_seeds[self.args.model_type]
        self.seed = seed
    
    def set_seed(self, repickle_args=True):
        
        # tf.keras.utils.set_random_seed(self.seed)
        tf.random.set_seed(self.seed)
        
        setattr(self.args, 'seed', self.seed)
        if repickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    def config_and_train_model(self, print_model_architecture = True, print_loss_over_epoch = True):
        
        args = self.args
        train_inputs, train_targets = self.train_dataset
        val_inputs, val_targets = self.val_dataset
        
        model = self.cls_constructor.create_model()
        if print_model_architecture:
            with open(f'{args.output_folder}/model_summary.txt', 'w') as f:
                model.build_graph().summary(print_fn=lambda x: f.write(x + '\n'))
        
        ## set up callbacks
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = args.reduce_LR_factor, patience = args.reduce_LR_patience, min_lr = 0.0000)
        callbacks = [reduce_lr]
        
        if args.use_ckpt:
            ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=args.ckpt_folder,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True)
            callbacks.append(ckpt)
        
        if args.ES_patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=args.ES_patience, monitor='val_loss',min_delta=0, mode='min')
            callbacks.append(early_stopping)
        if args.verbose > 0:
            loss_printer = LossPrintCallback(args.verbose)
            callbacks.append(loss_printer)
        
        ## compile
        print(f'[{args.model_type} model training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        model.compile(loss = self.criterion,
                      optimizer = Adam(learning_rate=args.learning_rate),
                      metrics = [tf.keras.metrics.RootMeanSquaredError()])
        ## train
        history = model.fit(train_inputs,
                            train_targets,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            validation_data=(val_inputs,val_targets),
                            callbacks=callbacks,
                            verbose=0)
        print(f'[{args.model_type} model training ends] epoch = {len(history.history["loss"])} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        if args.use_ckpt:
            model.load_weights(args.ckpt_folder)
        self.model = model
        
        if print_loss_over_epoch:
            plot_loss_over_epoch(history, args, save_as_file = f'{args.output_folder}/loss_over_epoch.png')
            
    def config_predictor(self):
        
        args =  self.args
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
        print_model_architecture = True,
        print_loss_over_epoch = True,
        print_train_err = True,
        plot_fitted = True,
        pickle_predictions = True,
        export_fcast_err_to_excel = True
    ):
    
        self.set_seed(repickle_args=repickle_args)
        self.source_data()
        self.generate_train_val_datasets()
        self.config_and_train_model(print_model_architecture=print_model_architecture, print_loss_over_epoch=print_loss_over_epoch)
        self.eval_training(print_train_err=print_train_err,plot_fitted=plot_fitted)
        self.config_predictor()
        
        self.run_forecast(pickle_predictions=pickle_predictions)
        self.eval_forecast(export_to_excel=export_fcast_err_to_excel)
