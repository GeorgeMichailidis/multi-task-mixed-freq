
"""
Multi-task Mixed Frequency Model
using Two LightGBMs, respectively for the low and the high frequency

(c) 2023, Jiahe Lin & George Michailidis
"""

import lightgbm as lgb
import numpy as np

class TwoGBM():
    """
    Two Light GBM that run time series regression for x and y, respectively:
    x depends on its own lags up to Lx
    y depends on its own lags (up to Ly) and the lags of x (up to Lx)
    """
    def __init__(
        self,
        dim_x,
        dim_y,
        hyper_params_x,
        hyper_params_y
    ):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hyper_params_x = hyper_params_x
        self.hyper_params_y = hyper_params_y
        
        self.x_models = []
        self.y_models = []

    def _flatten(self, inputs):
        """
        helper function for flattening the input
        argvs:
        - inputs: tuple with the first element corresponding to the lags of x and the second to the lags of y
        * inputs[0].shape = (batch_size, num_of_lags_for_high_freq, x_dim)
        * inputs[1].shape = (batch_size, num_of_lags_for_low_freq, y_dim)
        """
        sample_size = inputs[0].shape[0]
        assert inputs[1].shape[0] == sample_size
        
        x_flatten, y_flatten = inputs[0].reshape((sample_size,-1)), inputs[1].reshape((sample_size,-1))
        yx_flatten = np.concatenate((y_flatten, x_flatten),axis=-1)

        return x_flatten, yx_flatten
    
    def _train_single_regression(self, params, train_input, train_target, val_input = None, val_target = None):
        """
        training a single lgbm regression given input and target
        Argvs:
        - params: params used for model training
        - train_input/val_input: multivariate input (batch_size, input_dimension)
        - train_target/val_target: univariate target (batch_size, 1)
        """
        
        eval_results = {}
        lgb_trainset = lgb.Dataset(train_input, train_target)
        
        if (val_input is not None) and (val_target is not None):
            lgb_valset = lgb.Dataset(val_input, val_target)
            model = lgb.train(params,
                              lgb_trainset,
                              valid_sets=[lgb_trainset,lgb_valset],
                              valid_names=['train','val'],
                              callbacks=[lgb.record_evaluation(eval_results),
                                         lgb.early_stopping(params['early_stopping_round'])]
                              )
        else: ## no validation set
            model = lgb.train(params,
                              lgb_trainset,
                              callbacks=[lgb.record_evaluation(eval_results)])
            
        return model, eval_results
        
    def train(self, train_dataset, val_dataset = None, verbose=True):
        """
        Argvs:
        - train_dataset/val_dataset: tuple, with the first element corresponding to input (tuple) and the second to targets (tuple)
        Note:
        * inputs: tuple with the first element corresponding to the lags of x and the second to the lags of y
        * targets: tuple with the first element corresponding to the target of x and the second to the target of y
        """
        train_inputs, train_targets = train_dataset
        hf_train_input, lf_train_input = self._flatten(train_inputs)

        meta_evals = {}
        if val_dataset: ## in the case val_dataset is not None
            val_inputs, val_targets = val_dataset
            hf_val_input, lf_val_input = self._flatten(val_inputs)
            
            ## train for each coordinate of the hf variable
            for ix in range(self.dim_x):
                if verbose:
                    print(f'training lgbm for high frequency coordinate = {ix+1}/{self.dim_x}')
                model, meta_evals[f'x_{ix}'] = self._train_single_regression(self.hyper_params_x, hf_train_input, train_targets[0][:,ix], hf_val_input, val_targets[0][:,ix])
                self.x_models.append(model)
                
            ## train for each coordinate of the lf variable
            for iy in range(self.dim_y):
                if verbose:
                    print(f'training lgbm for low frequency coordinate = {iy+1}/{self.dim_y}')
                model, meta_evals[f'y_{iy}'] = self._train_single_regression(self.hyper_params_y, lf_train_input, train_targets[1][:,iy], lf_val_input, val_targets[1][:,iy])
                self.y_models.append(model)
        else:
            ## train for each coordinate of the hf variable
            for ix in range(self.dim_x):
                if verbose:
                    print(f'training lgbm for high frequency coordinate = {ix+1}/{self.dim_x}')
                
                model, meta_evals[f'x_{ix}'] = self._train_single_regression(self.hyper_params_x, hf_train_input, train_targets[0][:,ix])
                self.x_models.append(model)
            ## train for each coordinate of the lf variable
            for iy in range(self.dim_y):
                if verbose:
                    print(f'training lgbm for low frequency coordinate = {iy+1}/{self.dim_y}')
                model, meta_evals[f'y_{iy}'] = self._train_single_regression(self.hyper_params_y, lf_train_input, train_targets[1][:,iy])
                self.y_models.append(model)
            
        return meta_evals
    
    def predict(self, eval_dataset):
        """
        Argvs:
        - eval_dataset: dataset for direct evaluation; a tuple with the first element corresponding to the lags of x and the second to the lags of y
        """
        hf_input, lf_input = self._flatten(eval_dataset)
        
        x_predictions = []
        for ix in range(self.dim_x):
            x_out = self.x_models[ix].predict(hf_input)
            x_predictions.append(x_out)
        
        y_predictions = []
        for iy in range(self.dim_y):
            y_out = self.y_models[iy].predict(lf_input)
            y_predictions.append(y_out)
            
        x_predictions, y_predictions = np.array(x_predictions).transpose(), np.array(y_predictions).transpose()
        return x_predictions, y_predictions
    
class GBMPred():
    def __init__(
        self,
        model,
        freq_ratio,
        scaler_x,
        scaler_y,
        apply_inv_scaler = True
    ):
        self.x_models = model.x_models
        self.y_models = model.y_models
        self.freq_ratio = freq_ratio
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.apply_inv_scaler = apply_inv_scaler
        
    def forecast_one_step(self, models, x):
        """
        forecast one-step-ahead, can be used for either lf or hf
        Argv:
            models: list of trained lgb models, of the same length as the hf or lf dimension
            x: (2D tensor) lag input, of shape (1, num_of_lags, input_dim)
        Return:
            forecast_vals: one-step-ahead forecast, of shape (1, output_dim) where output_dim == either hf or lf variable dimension
        """
        assert x.shape[0] == 1
        x_flat = x.reshape((x.shape[0],-1))
        forecast_vals = []
        for model in models:
            prediction = model.predict(x_flat)[0]
            forecast_vals.append(prediction)
        return np.array(forecast_vals).reshape((1,-1))
        
    def forecast_hf_multi_step(self, x_input, num_steps=1):
        """
        forecast multi-step-ahead for x
        Argv:
            x_input: (tensor) lag input, of shape (1, Lx, dim_x)
        Return:
            x_pred_vals:multi-step-ahead forecast, (num_steps, dim_x)
        """
        x_pred_vals = []
        for step in range(num_steps):
            x_forecast = self.forecast_one_step(self.x_models, x_input)
            x_pred_vals.append(np.squeeze(x_forecast))
            x_input_2d = np.concatenate((x_input[0,1:,:],x_forecast),axis=0)
            x_input = np.expand_dims(x_input_2d, axis=0)
        
        x_pred_vals = np.array(x_pred_vals)
        return x_pred_vals
    
    def predict_system_one_cycle(self,inputs,x_step=0):
        """
        forecast the system, where x_step corresponds to the number of steps of x "into" the quarter
        Argv:
            inputs = (x, y): (tensor) encoder / decoder input, (1, Lx, dim_x), (1, Ty, dim_y)
            x_step: (int) number of steps that x is alread into the cycle
        Ret:
            x_pred: prediction of x, shape = (freq_ratio, dim_x)
            y_pred: prediction of y, shape = (dim_y, )
        """
        
        assert x_step <= self.freq_ratio
        x_steps_to_forecast = self.freq_ratio - x_step
        
        x_input, y_input = inputs[0], inputs[1]
        if x_steps_to_forecast < 1:
            pass
        else:
            x_pred_vals = self.forecast_hf_multi_step(x_input, num_steps=x_steps_to_forecast)
            ## arrange input
            x_input_2d = np.concatenate((x_input[0,x_steps_to_forecast:,:],x_pred_vals),axis=0)
            x_input = np.expand_dims(x_input_2d,axis=0)
        
        x_flat = x_input.reshape((x_input.shape[0],-1))
        y_flat = y_input.reshape((y_input.shape[0],-1))
        yx = np.concatenate((y_flat, x_flat),axis=-1)
    
        ## organize predictions
        y_pred = self.forecast_one_step(self.y_models, yx)
        x_pred = x_input[0,-self.freq_ratio:,:]
        
        ## get prepared for the next cycle input
        x_input_nxt = x_input
        y_input_nxt = np.concatenate([y_input[:,1:,:],np.expand_dims(y_pred,axis=0)],axis=1)
        
        return (x_pred, y_pred[0]), (x_input_nxt, y_input_nxt)
        
    def __call__(self, inputs, x_step, horizon=4):
        assert x_step <= self.freq_ratio
        
        x_pred, y_pred = [], []
        for hz in range(horizon):
            predictions, inputs = self.predict_system_one_cycle(inputs, x_step = x_step)
            x_pred.append(predictions[0])
            y_pred.append(predictions[1])
            ## reset x_step
            x_step = 0
            
        x_pred, y_pred = np.concatenate(x_pred, axis=0), np.stack(y_pred, axis=0)
        if self.apply_inv_scaler:
            x_pred = self.scaler_x.inverse_transform(x_pred)
            y_pred = self.scaler_y.inverse_transform(y_pred)
        return x_pred, y_pred
