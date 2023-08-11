"""
Multi-task Mixed Frequency Model based on a seq2seq architecture
(c) 2023 Jiahe Lin & George Michailidis
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Dot, Input, LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import RepeatVector, Dense, Activation
from tensorflow.keras.models import Model

from models import _baseSeqPred

class PreAttnEncoder(tf.keras.Model):
    """Pre-attention Encoder module"""
    def __init__(self, dim_x, n_a, dropout_rate = 0.2, bidirectional_encoder=False):
        """
        dim_x: (int) dimension of the (encoder) input high-frequency sequence
        n_a: (int) hidden state dimension of the pre-attention LSTM
        dropout_rate: (float) dropout rate
        """
        super(PreAttnEncoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        if self.bidirectional_encoder:
            self.LSTM = Bidirectional(LSTM(units = n_a, return_sequences = True, name = 'pre_attn_encoder'))
        else:
            self.LSTM = LSTM(units = n_a, return_sequences = True, name = 'pre_attn_encoder')
            
    def call(self, x, training=False):
        """
        forward pass for the encoder
        Argv:
            x: (tensor) shape (batch_size, Lx, dim_x)
            training: (bool) flag for whether in training mode
        Return:
            a: (tensor) sequence of LSTM hiddne states, (batch_size, Lx, n_a)
        """
        a = self.LSTM(x)
        return a

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

class MTMFSeq2Seq(tf.keras.Model):
    def __init__(
        self,
        dim_x,
        dim_y,
        Lx,
        Tx,
        Ty,
        n_a,
        n_s,
        n_align_x,
        n_align_y,
        fc_x,
        fc_y,
        dropout_rate,
        freq_ratio = 3,
        bidirectional_encoder = False,
        l1reg = 1e-5,
        l2reg = 1e-4
    ):
        """
        dim_x: (int) dimension of the (encoder) input high-frequency sequence
        dim_y: (int) dimension of the (decoder) input/output low-frequency sequence
        Lx: (int) length of the input high-frequency sequence (encoder)
        Tx: (int) lenght of the target high-frequency sequence (decoder)
        Ty: (int) length of the output low-frequency sequence (decoder)
        n_a: (int) hidden state dimension of the pre-attention/post-atten LSTM for x
        n_s: (int) hidden state dimension of the post-attention LSTM
        n_align_{x,y}: (int) hidden state dimension of the dense layer in the alignment model
        fc_{x,y}: (int) hidden state dimension of the dense layer before the dropout layer
        dropout_rate: (float) dropout rate for the layer before output
        freq_ratio: (int) frequency ratio
        bidirectional_encoder: (bool) whether the encoder should be bidirectional
        """
        
        super(MTMFSeq2Seq, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.Lx = Lx
        self.Tx = Tx
        self.Ty = Ty
        self.n_a = n_a
        self.n_s = n_s
        self.n_align_x = n_align_x
        self.n_align_y = n_align_y
        self.fc_x = fc_x
        self.fc_y = fc_y
        
        self.freq_ratio = freq_ratio
        self.bidirectional_encoder = bidirectional_encoder
        
        self.l1reg = l1reg
        self.l2reg = l2reg
        
        ## for the encoder
        self.pre_attn = PreAttnEncoder(dim_x, n_a, dropout_rate = dropout_rate, bidirectional_encoder = bidirectional_encoder)
        ## for the attention alignment model
        self.one_step_attention_x = OneStepAttn(n_align_x)
        self.one_step_attention_y = OneStepAttn(n_align_y)
        
        ## for the xdecoder
        self.post_attn_x = LSTM(n_a,return_state=True,name='post_attn_decoder_x')
        self.ffn1_x = Dense(fc_x,activation='relu')
        self.dropout_fn_x = Dropout(dropout_rate)
        self.ffn2_x = Dense(dim_x,kernel_regularizer = regularizers.L1L2(l1=self.l1reg, l2=self.l2reg))
        
        ## for the ydecoder
        self.post_attn_y = LSTM(n_s,return_state=True,name='post_attn_decoder_y')
        self.ffn1_y = Dense(fc_y,activation='relu')
        self.dropout_fn_y = Dropout(dropout_rate)
        self.ffn2_y = Dense(dim_y, kernel_regularizer = regularizers.L1L2(l1=self.l1reg, l2=self.l2reg))
        
    def initialize_state(self, batch_size, dim):
        if batch_size is not None:
            return tf.zeros((batch_size, dim))
        else:
            return tf.Variable((np.empty((0,dim),dtype=np.float32)),shape=[None,dim])
    
    def call(self, batch_inputs, training = False):
        
        x_encoder_in, x_decoder_in, y_decoder_in = batch_inputs
        batch_size = tf.shape(x_encoder_in)[0]
        
        ##############################################
        ## stage 1: pre-attn encoding
        ##############################################
        a = self.pre_attn(x_encoder_in, training=training)
        
        ##############################################
        ## stage 2.1: x -> y decoding
        ##############################################
        s_y, c_y = self.initialize_state(batch_size, self.n_s), self.initialize_state(batch_size, self.n_s)
        for t in range(self.Ty):
            a_idx = int((t+1)*self.freq_ratio-1)
            a_to_attend = a[:,(a_idx-self.freq_ratio+1):(a_idx+1),:]
            context = self.one_step_attention_y([a_to_attend,s_y])
            post_attn_input = Concatenate(axis=-1)([context,tf.expand_dims(y_decoder_in[:,t,:],axis=1)])
            s_y, _, c_y = self.post_attn_y(inputs = post_attn_input, initial_state = [s_y,c_y])
        
        y_pred = self.ffn1_y(s_y)
        y_pred = self.dropout_fn_y(y_pred, training=training)
        y_pred = self.ffn2_y(y_pred)
        
        ##############################################
        ## stage 2.2: x -> x decoding
        ##############################################
        s_x = a[:,-1,:self.n_a]
        c_x = self.initialize_state(batch_size, self.n_a)
        
        x_pred_by_step = []
        for t in range(x_decoder_in.shape[1]):
            context = self.one_step_attention_x([a,s_x])
            post_attn_input = Concatenate(axis=-1)([context,tf.expand_dims(x_decoder_in[:,t,:],axis=1)])
            s_x, _, c_x = self.post_attn_x(inputs = post_attn_input, initial_state = [s_x,c_x])
            x_pred_curr = self.ffn1_x(s_x)
            x_pred_curr = self.dropout_fn_x(x_pred_curr, training=training)
            x_pred_curr = self.ffn2_x(x_pred_curr)
            x_pred_by_step.append(x_pred_curr)
        x_pred = tf.stack(x_pred_by_step, axis=1)
        
        return [x_pred, y_pred]

    def build_graph(self):
        x_encoder_in = tf.keras.layers.Input(shape=(self.Lx,self.dim_x))
        x_decoder_in = tf.keras.layers.Input(shape=(self.Tx,self.dim_x))
        y_decoder_in = tf.keras.layers.Input(shape=(self.Ty,self.dim_y))
        inputs = [x_encoder_in, x_decoder_in, y_decoder_in]
        outputs = self.call(inputs)
        return Model(inputs=inputs, outputs=outputs)

class MTMFSeq2SeqPred(_baseSeqPred):
    
    def __init__(
        self,
        model,
        scaler_x,
        scaler_y,
        apply_inv_scaler=True
    ):
        super().__init__(model, scaler_x, scaler_y, apply_inv_scaler)
