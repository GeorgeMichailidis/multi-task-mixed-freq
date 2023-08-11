"""
Multi-task Mixed Frequency Model based on a seq2one architecture
(c) 2023 Jiahe Lin & George Michailidis
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Dot, Input, LSTM, Dropout
from tensorflow.keras.layers import RepeatVector, Dense, Activation
from tensorflow.keras.models import Model

class PreAttnEncoder(tf.keras.Model):
    """Pre-attention Encoder module"""
    def __init__(self, dim_x, fc_dim, n_a, dropout_rate=0.2, bidirectional_encoder=False, l1reg = 1e-5, l2reg = 1e-4):
        """
        dim_x: (int) dimension of the (encoder) input high-frequency sequence
        n_a: (int) hidden state dimension of the pre-attention LSTM
        dropout_rate: (float) dropout rate
        """
        super(PreAttnEncoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        self.l1reg = l1reg
        self.l2reg = l2reg
        
        if self.bidirectional_encoder:
            self.LSTM = Bidirectional(LSTM(units = n_a, return_sequences = True, name = 'pre_attn_encoder'))
        else:
            self.LSTM = LSTM(units = n_a, return_sequences = True, name = 'pre_attn_encoder')
        
        self.ffn1 = Dense(fc_dim,activation='relu')
        self.dropout_fn = Dropout(dropout_rate)
        self.ffn2 = Dense(dim_x,activation='linear',name='x_output',kernel_regularizer = regularizers.L1L2(l1=self.l1reg, l2=self.l2reg))
        
    def call(self, x, training=False):
        """
        forward pass for the encoder
        Argv:
            x: (tensor) shape (batch_size, Lx, dim_x)
            training: (bool) flag for whether in training mode
        Return:
            x_pred: (tensor), the next step prediction for x (batch_size, dim_x)
            a: (tensor) sequence of LSTM hiddne states, (batch_size, Lx, n_a)
        """
        a = self.LSTM(x)
        
        x_pred = self.ffn1(a[:,-1,:])
        x_pred = self.dropout_fn(x_pred, training=training)
        x_pred = self.ffn2(x_pred)
        
        return x_pred, a

class OneStepAttn(tf.keras.layers.Layer):
    """ Attention alignment module"""
    def __init__(self, n_align):
        """
        n_align: (int) hidden unit of the alignment model
        """
        super(OneStepAttn, self).__init__()
        self.densor1 = Dense(n_align, activation = "tanh")
        self.densor2 = Dense(1, activation = "relu")
        self.activator = Activation(OneStepAttn._softmax, name='attention_weights')

    @staticmethod
    def _softmax(x, axis=1):
        """
        Customized softmax function that is suitable for getting attn
        Argv:
            x : Tensor.
            axis: Integer, axis along which the softmax normalization is applied.
        Returns
            Tensor, output of softmax transformation.
        """
        ndim = K.ndim(x)
        if ndim == 2: return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

    def call(self, attn_input):
        """
        forward pass: performs one step customized attention that outputs a context vector computed as a dot product of the attention weights "alphas" and the hidden states "a" from the LSTM encoder.
        Argv:
            a: hidden state from the pre-attention LSTM, shape = (m, *, n_a)
            s_prev: previous hidden state of the (post-attn) LSTM, shape = (m, n_s)
        Returns:
            context: context vector, input of the next (post-attention) LSTM cell
        """
        a, s_prev = attn_input
        s_prev = RepeatVector(a.shape[1])(s_prev) #(m, a.shape[1], n_s)
        concat = Concatenate(axis=-1)([a,s_prev])
        e = self.densor1(concat)
        energies = self.densor2(e)
        alphas = self.activator(energies)
        context = Dot(axes = 1)([alphas,a])
        return context

class MTMFSeq2One(tf.keras.Model):
    def __init__(
        self,
        Lx,
        dim_x,
        Ty,
        dim_y,
        n_a,
        n_s,
        n_align,
        fc_x,
        fc_y,
        dropout_rate,
        freq_ratio = 3,
        bidirectional_encoder = False,
        l1reg = 1e-5,
        l2reg = 1e-4
    ):
        """
        Lx: (int) length of the input high-frequency sequence (encoder_steps)
        dim_x: (int) dimension of the (encoder) input high-frequency sequence
        Ty: (int) length of the output low-frequency sequence (decoder_steps)
        dim_y: (int) dimension of the (decoder) input/output low-frequency sequence
        n_a: (int) hidden state dimension of the pre-attention LSTM
        n_s: (int) hidden state dimension of the post-attention LSTM
        n_align: (int) hidden state dimension of the dense layer in the alignment model
        fc_x, fc_y: (int) hidden state dimension before the dropout layer
        dropout_rate: (float) dropout rate for the layer before output
        freq_ratio: (int) frequency ratio
        """
        super(MTMFSeq2One, self).__init__()
        self.Lx = Lx
        self.Ty = Ty
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_a = n_a
        self.n_s = n_s
        self.n_align = n_align
        self.fc_x = fc_x
        self.fc_y = fc_y
        self.freq_ratio = freq_ratio
        self.bidirectional_encoder = bidirectional_encoder
        self.l1reg = l1reg
        self.l2reg = l2reg
        
        ## for the encoder
        self.pre_attn = PreAttnEncoder(dim_x, n_a, fc_x, dropout_rate = dropout_rate, bidirectional_encoder = bidirectional_encoder, l1reg = self.l1reg, l2reg = self.l2reg)
        ## for the attention alignment model
        self.one_step_attention = OneStepAttn(n_align)
        ## for the decoder
        self.post_attn = LSTM(n_s,return_state=True,name='post_attn_decoder')
        self.ffn1 = Dense(fc_y, activation='relu')
        self.dropout_fn = Dropout(dropout_rate)
        self.ffn2 = Dense(dim_y, name='y_output',kernel_regularizer = regularizers.L1L2(l1=self.l1reg, l2=self.l2reg))
    
    def initialize_state(self, batch_size, dim):
        if batch_size is not None:
            return tf.zeros((batch_size, dim))
        else:
            return tf.Variable((np.empty((0,dim),dtype=np.float32)),shape=[None,dim])
    
    def call(self, batch_inputs, training = False):
        """forward pass"""
        x, y = batch_inputs ## encoder input, decoder input
        x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
        batch_size = tf.shape(x)[0]
        ##############################################
        ## stage 1: pre-attn encoding
        ##############################################
        x_pred, a = self.pre_attn(x, training=training)
        ##############################################
        ## stage 2: attention-based decoding
        ##############################################
        s = self.initialize_state(batch_size, self.n_s)
        c = self.initialize_state(batch_size, self.n_s)
        for t in range(self.Ty):
            a_idx = int((t+1)*self.freq_ratio-1)
            a_to_attend = a[:,(a_idx-self.freq_ratio+1):(a_idx+1),:]
            context = self.one_step_attention([a_to_attend,s])
            ## context + teacher forcing, post-attention LSTM input
            post_attn_input = Concatenate(axis=-1)([context,tf.expand_dims(y[:,t,:],axis=1)])
            # post-attention LSTM cell to the "context" vector
            s, _, c = self.post_attn(inputs = post_attn_input, initial_state = [s,c])
        
        y_pred = self.ffn1(s)
        y_pred = self.dropout_fn(y_pred, training=training)
        y_pred = self.ffn2(y_pred)
        return [x_pred, y_pred]

    def build_graph(self):
        x_input = Input(shape=(self.Lx,self.dim_x))
        y_input = Input(shape=(self.Ty,self.dim_y))
        return Model(inputs=[x_input,y_input], outputs=self.call([x_input,y_input]))

class MTMFSeq2OnePred(tf.Module):
    def __init__(self, multi_task_mf_model,scaler_x,scaler_y,apply_inv_scaler=True):
        self.model = multi_task_mf_model
        self.freq_ratio = self.model.freq_ratio
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.apply_inv_scaler = apply_inv_scaler
    
    def forecast_hf_one_step(self, x):
        """
        forecast one-step-ahead for x
        Argv:
            x: (3D tensor) encoder input, of shape (1, Lx, dim_x)
        Ret:
            x_forecast: one-step-ahead forecast, (1, dim_x)
        """
        assert x.shape[1] == self.model.Lx, 'Lx implied by the input x does not match the number of encoder_steps'
        x_forecast, _ = self.model.pre_attn(x,training=False)
        return x_forecast
    
    def forecast_hf_multi_step(self, x_input, num_steps=1):
        """
        forecast multi-step-ahead for x
        Argv:
            x_input: (tensor) encoder input, of shape [1, Lx, dim_x]
        Ret:
            x_pred_vals: multi-step-ahead forecast, (num_steps, dim_x)
        """
        x_pred_vals = []
        for step in range(num_steps):
            x_forecast = self.forecast_hf_one_step(x_input)
            ## record forecast
            x_pred_vals.append(np.squeeze(x_forecast))
            ## update input
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
        
        Example for x_step with monthly/quarterly data
        x_step = 0: both x and y have up to December data available
        x_step = 1: x has up to Jan data, y has up to Dec data, nowcast Feb, March for x and March for y
        x_step = 2: x has up to Feb data, y has up to Dec data, nowcast March for x and March for y
        x_step = 3: x has up to March data, y has up to Dec data, nowcast March for y
        """
        assert x_step <= self.model.freq_ratio
        x_steps_to_forecast = self.freq_ratio - x_step
        
        x_input, y_input = inputs[0], inputs[1]
        if x_steps_to_forecast < 1: ## directly make a nowcast
            pass
        else:
            x_pred_vals = self.forecast_hf_multi_step(x_input, num_steps = x_steps_to_forecast)
            ## arrange input
            x_input_2d = np.concatenate((x_input[0,x_steps_to_forecast:,:],x_pred_vals),axis=0)
            x_input = np.expand_dims(x_input_2d,axis=0)
        
        ## organize predictions
        _, y_pred = self.model.predict([x_input, y_input])
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
