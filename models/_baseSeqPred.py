"""
Multi-task Mixed Frequency Model
(c) 2023 Jiahe Lin & George Michailidis
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Dot, Input, LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import RepeatVector, Dense, Activation
from tensorflow.keras.models import Model

class _baseSeqPred(tf.Module):
    '''
    base class for predictor that generate forecast/nowcasts, where the output is multi-step
    this is shared between seq2seq and transformer
    '''
    def __init__(self, model,scaler_x, scaler_y, apply_inv_scaler=True):
        
        self.model = model
        self.freq_ratio = self.model.freq_ratio
        self.Tx = self.model.Tx
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.apply_inv_scaler = apply_inv_scaler
    
    def forecast_hf_multi_step(self, inputs, num_of_steps):
        
        x_encoder_in, x_decoder_in, y_decoder_in = inputs
        assert x_decoder_in.shape[1] - 1 + num_of_steps == self.freq_ratio
        
        x_pred_vals = []
        for step in range(num_of_steps):
            x_forecast_multi, _ = self.model([x_encoder_in, x_decoder_in, y_decoder_in],training=False)
            x_forecast = x_forecast_multi[:,-1,:]
            x_pred_vals.append(np.squeeze(x_forecast))
            x_decoder_in = np.concatenate([x_decoder_in, np.expand_dims(x_forecast,axis=1)],axis=1)
        
        x_pred_vals = np.array(x_pred_vals)
        return x_pred_vals ## (num_of_steps, dim_x)
        
    def predict_system_one_cycle(self, inputs, x_step):
        
        x_encoder_in, x_decoder_in, y_decoder_in = inputs
        assert x_step == x_decoder_in.shape[1] - 1
        x_steps_to_forecast = self.freq_ratio - x_step
        
        if x_steps_to_forecast < 1: ## directly make a nowcast
            x_decoder_in_aug = x_decoder_in
        else:
            x_pred_vals = self.forecast_hf_multi_step(inputs, num_of_steps = x_steps_to_forecast)
            x_decoder_in_aug = np.concatenate([x_decoder_in, np.expand_dims(x_pred_vals,axis=0)],axis=1) ## (1, Tx+1, dim_x)
        
        ## collect x prediction
        x_pred = x_decoder_in_aug[0,-self.freq_ratio:,:]
        ## update x_encoder_in & decoder
        x_encoder_in = np.concatenate([x_encoder_in[:,self.freq_ratio:,:], x_decoder_in_aug[:,1:,:]],axis=1)
        x_decoder_in = np.expand_dims(x_decoder_in_aug[:,-1,:], axis=1)
        
        ## make a nowcast of y
        _, y_pred = self.model([x_encoder_in, x_decoder_in, y_decoder_in], training=False)
        ## update of y decoder
        y_decoder_in = np.concatenate([y_decoder_in[:,1:,:], np.expand_dims(y_pred,axis=0)], axis=1)
        
        return (x_pred, y_pred[0]), (x_encoder_in, x_decoder_in, y_decoder_in)
        
    def __call__(self, inputs, x_step, horizon = 4):
        
        assert x_step == inputs[1].shape[1] - 1
        assert x_step <= self.freq_ratio
        
        x_pred, y_pred = [], []
        for hz in range(horizon):
            predictions, inputs = self.predict_system_one_cycle(inputs, x_step=x_step)
            x_pred.append(predictions[0])
            y_pred.append(predictions[1])
            x_step = 0
        x_pred, y_pred = np.concatenate(x_pred, axis=0), np.stack(y_pred, axis=0)
        if self.apply_inv_scaler:
            x_pred = self.scaler_x.inverse_transform(x_pred)
            y_pred = self.scaler_y.inverse_transform(y_pred)
        return x_pred, y_pred
