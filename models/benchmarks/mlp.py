"""
Multi-task Mixed Frequency Model with an MLP-based architecture
(c) 2023 Jiahe Lin & George Michailidis
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Dot, Input, Dropout, Dense, Add, BatchNormalization, Flatten
from tensorflow.keras.models import Model

class FFBlock(tf.keras.layers.Layer):
    """
    Feed-forward block with the following layers:
    linear -> activation -> dropout -> linear (output)
    """
    def __init__(
        self,
        hidden_dim,
        output_dim,
        activation='relu',
        dropout_rate=0.1,
    ):
        super(FFBlock, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fflayers = tf.keras.Sequential([
                Dense(hidden_dim,activation=activation),
                Dropout(dropout_rate),
                Dense(output_dim)
            ])
            
    def call(self, x, training=False):
        output = self.fflayers(x,training=training)
        return output
        
class FFNet(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        activation='relu',
        architect='stack',
        dropout_rate=0.1
    ):
        super(FFNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.architect = architect
        self.bn_x = BatchNormalization() ## batch normalization layer for input
        
        self.fc1 = FFBlock(self.hidden_dim, output_dim=self.input_dim, activation=activation, dropout_rate=dropout_rate)
        self.bn1 = BatchNormalization()
        self.fc2 = FFBlock(self.hidden_dim, output_dim=self.input_dim, activation=activation, dropout_rate=dropout_rate)
        self.bn2 = BatchNormalization()
                    
        self.output_layer = FFBlock(self.hidden_dim, output_dim=self.output_dim, activation=activation, dropout_rate=dropout_rate)
    
    def call(self, x, training=False):
        """
        forward pass
        x (tensor): of shape (batch_size, self.input_dim)
        """
        assert x.shape[1] == self.input_dim
        x = self.bn_x(x,training=training)
        
        if self.architect == 'concat':
            ## resnet1
            y1 = self.fc1(x,training=training)
            y1 = Add()([x,y1])
            y1 = self.bn1(y1)
            ## resnet2
            y2 = self.fc2(x,training=training)
            y2 = Add()([x,y2])
            y2 = self.bn2(y2)
            ## concat
            y = Concatenate()([y1,y2])
        elif self.architect == 'stack':
            y = self.fc1(x,training=training)
            y = Add()([x,y])
            y = self.bn1(y)
            y = self.fc2(y,training=training)
            y = Add()([x,y])
            y = self.bn2(y)
        
        output = self.output_layer(y,training=training)
        return output
        
class TwoMLP(tf.keras.Model):
    """
    Two MLPs that predict x and y resp:
    x depends on its own lags up to Lx
    y depends on its own lags (up to Ly) and the lags of x (up to Lx)
    """
    def __init__(
        self,
        dim_x,
        dim_y,
        Lx,
        Ly,
        hidden_dim_x,
        activation_x,
        architect_x,
        hidden_dim_y,
        activation_y,
        architect_y,
        dropout_rate = 0.1
    ):
        super(TwoMLP, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.Lx = Lx
        self.Ly = Ly
        self.hidden_dim_x = hidden_dim_x
        self.hidden_dim_y = hidden_dim_y
        self.activation_x = activation_x
        self.activation_y = activation_y
        self.architect_x = architect_x
        self.architect_y = architect_y
        self.flatx = Flatten()
        self.flaty = Flatten()
        self.x_net = FFNet(input_dim=self.dim_x*self.Lx, output_dim = self.dim_x, hidden_dim = hidden_dim_x, activation = activation_x, dropout_rate = dropout_rate)
        self.y_net = FFNet(input_dim=dim_x*Lx + dim_y*Ly, output_dim = dim_y, hidden_dim = hidden_dim_y, activation = activation_y, dropout_rate=dropout_rate)
        
    def call(self, batch_inputs, training = False):
        """forward pass"""
        x_lags, y_lags = batch_inputs
        
        ## flatten
        x_lags_flat = self.flatx(x_lags)
        y_lags_flat = self.flaty(y_lags)
        yx_lags_flat = Concatenate(axis=-1)([y_lags_flat,x_lags_flat])
        
        x_pred = self.x_net(x_lags_flat, training=training)
        y_pred = self.y_net(yx_lags_flat, training=training)
        return [x_pred, y_pred]

    def build_graph(self):
        x_input = Input(shape=(self.Lx,self.dim_x))
        y_input = Input(shape=(self.Ly,self.dim_y))
        return Model(inputs=[x_input,y_input], outputs=self.call([x_input,y_input]))

class MLPPred(tf.Module):
    def __init__(
        self,
        two_mlp_model,
        freq_ratio,
        scaler_x,
        scaler_y,
        apply_inv_scaler=True
    ):
        self.model = two_mlp_model
        self.freq_ratio = freq_ratio
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.apply_inv_scaler = apply_inv_scaler
        
    def forecast_one_step(self, model, x):
        """
        forecast one-step-ahead
        Argv:
            x: (2D tensor) lag input, of shape (1, num_of_lags, input_dim)
        Ret:
            x_forecast: one-step-ahead forecast, of shape (1, output_dim)
        """
        assert x.shape[0] == 1
        x_flat = Flatten()(x)
        x_forecast = model(x_flat,training=False)
        
        return x_forecast
    
    def forecast_hf_multi_step(self, x_input, num_steps=1):
        """
        forecast multi-step-ahead for x
        Argv:
            x_input: (tensor) lag input, of shape (1, Lx, dim_x)
        Ret:
            x_pred_vals: multi-step-ahead forecast, (num_steps, dim_x)
        """
        x_pred_vals = []
        for step in range(num_steps):
            x_forecast = self.forecast_one_step(self.model.x_net, x_input)
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
        
        x_flat = Flatten()(x_input)
        y_flat = Flatten()(y_input)
        yx = tf.concat([y_flat,x_flat],axis=1)
        
        ## organize predictions
        y_pred = self.forecast_one_step(self.model.y_net, yx)
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
