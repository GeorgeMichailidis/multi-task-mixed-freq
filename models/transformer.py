"""
Multi-Task Mixed Frequency Model
based on a transformer encoder-decoder architecture
(c) 2023 Jiahe Lin & George Michailidis
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Dot, Input, LSTM, Dropout, GRU
from tensorflow.keras.layers import RepeatVector, Dense, Activation
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model

from models import _baseSeqPred

class PositionalEncoding():
    def __init__(self,sequence_length,dim):
        '''
            sequence_length (int) -- length of the sequence whose positional encoding needs to be obtained
            dim (int) -- embedding dimension
        '''
        self.sequence_length = sequence_length
        self.dim = dim
    
    def get_angles(self):
        '''
        Return: angles -- numpy array of shape (sequence_length,dim)
        '''
        pos_vec = np.arange(self.sequence_length)[:,np.newaxis] #column vec containiner [[0],[1],...,[N-1]]
        coordinate_vec = np.arange(self.dim)[np.newaxis,:] #row vec containing the dimension coordinates [[0,1,...,d-1]]
        return pos_vec / np.power(10000, (2 * (coordinate_vec//2)) / np.float32(self.dim))
        
    def get_positional_encoding(self):
        '''
        Return: pos_encoding -- (sequence_length, dim) tensor with the position encodings
        '''
        angle_rads = self.get_angles()
        ## apply sin to even indices and cos to odd indices
        angle_rads[:, 0::2], angle_rads[:, 1::2] = np.sin(angle_rads[:, 0::2]), np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads,dtype=tf.float32)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim_x, key_dim, fc_dim, num_heads = 2, dropout_rate = 0.1, layernorm_eps = 1e-6):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads = num_heads,key_dim = key_dim, dropout = dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon = layernorm_eps)
        self.ffn1 = Dense(fc_dim, activation='relu')
        self.dropout_fn = Dropout(dropout_rate)
        self.ffn2 = Dense(dim_x)
          
    def call(self, x, training, look_ahead_mask = None):
        """
        Argv:
            x -- Tensor of shape (batch_size, Lx, dim_x)
            training -- Boolean, set to true to activate the training mode for dropout layers
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, Lx, dim_x)
        """
        attn_output = self.mha(x,x,x,look_ahead_mask) # self-attention (batch_size, Lx, dim_x)
        out = self.layernorm1(attn_output + x)  # (batch_size, Lx, dim_x)
        
        out = self.ffn1(out)
        out = self.dropout_fn(out, training=training)
        encoder_layer_out = self.ffn2(out)  # (batch_size, Lx, dim_x)
        
        return encoder_layer_out

class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder module: two multi-head attention blocks (self attention + encoder-decoder attention) followed by a fully connected block
    """
    def __init__(self, output_dim, key_dim, fc_dim, num_heads = 2, dropout_rate = 0.1, layernorm_eps = 1e-6):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
        
        self.ffn = tf.keras.Sequential([
                        Dense(fc_dim, activation='relu'),
                        Dense(output_dim)
                ])
                        
    def call(self, x, enc_output, training, look_ahead_mask):
        """
        Argv:
            x -- Tensor of shape (batch_size, decoder_seq_length, output_dim)
            enc_output --  Tensor of shape(batch_size, encoder_seq_length, encoder_dim)
            training -- Boolean, set to true to activate the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
        Returns:
            dec_layer_out -- Tensor of shape (batch_size, decoder_seq_length, output_dim)
            attn_w1 -- Tensor of shape (batch_size, num_heads, decoder_seq_length, decoder_seq_length)
            attn_w2 -- Tensor of shape (batch_size, num_heads, decoder_seq_length, encoder_seq_length)
        """
        ## self-attention for the target sequence
        attn1, attn_w1 = self.mha1(x, x, x, look_ahead_mask, return_attention_scores=True)  # (batch_size, decoder_seq_length, dim_x)
        attn1 = self.dropout1(attn1,training=training)
        out1 = self.layernorm1(attn1+x)
        
        ## enc-dec attention
        attn2, attn_w2 = self.mha2(out1, enc_output, enc_output, return_attention_scores=True)  # (batch_size, decoder_seq_length, dim_x)
        attn2 = self.dropout2(attn2,training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ## ff
        out_ffn = self.ffn(out2)
        out_ffn = self.dropout3(out_ffn)
        
        dec_layer_out = self.layernorm3(out_ffn + out2)
        return dec_layer_out, attn_w1, attn_w2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim_x, Lx, key_dim, fc_dim, num_layers = 1, num_heads = 2, dropout_rate = 0.1, layernorm_eps = 1e-6):
        super(Encoder, self).__init__()
        self.dim_x = dim_x
        self.Lx = Lx
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(self.Lx,self.dim_x).get_positional_encoding()
        self.enc_layers = [EncoderLayer(dim_x = self.dim_x, key_dim = key_dim, fc_dim = fc_dim, num_heads = num_heads, dropout_rate = dropout_rate,layernorm_eps = layernorm_eps) for _ in range(self.num_layers)]
        
    def call(self, x, look_ahead_mask=None, training = False):
        """
        Argv:
            x -- Tensor of shape (batch_size, Lx, dim_x)
            training -- Boolean, set to true to activate the training mode for dropout layers
        Returns:
            x -- Tensor of shape (batch_size, Lx, dim_x)
        """
        x *= tf.math.sqrt(tf.cast(self.dim_x,tf.float32))
        x = x + self.pos_encoding
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, look_ahead_mask)
        return x  # (batch_size, encoder_seq_length, dim_x)

class xDecoder(tf.keras.layers.Layer):
    def __init__(self, dim_x, Tx, key_dim, fc_dim, ffn_dim, num_layers = 1, num_heads = 2, dropout_rate=0.1, layernorm_eps=1e-6):
        super(xDecoder, self).__init__()
        self.dim_x = dim_x
        self.Tx = Tx
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(self.Tx, self.dim_x).get_positional_encoding()
        self.dec_layers = [DecoderLayer(output_dim=self.dim_x, key_dim=key_dim, fc_dim=fc_dim, num_heads=num_heads, dropout_rate=dropout_rate,layernorm_eps=layernorm_eps) for _ in range(self.num_layers)]
        ## finally feed-forward block
        self.ffn = tf.keras.Sequential([
                        Dense(ffn_dim, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(self.dim_x)
                    ])
        
    def call(self, x, enc_output, look_ahead_mask, training = False):
        """
        Forward  pass for the Decoder - high freq variable
        Argv:
            x -- Tensor of shape (batch_size, Tx, dim_x)
            enc_output --  Tensor of shape(batch_size, Lx, dim_x)
            training -- Boolean, set to true to activate the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
        Returns:
            x -- Tensor of shape (batch_size, Tx, dim_x)
            attention_weights - Dictionary of tensors containing all the attention weights, (batch_size, num_heads, Tx, Tx) or (batch_size, num_heads, Tx, Lx)
        """
        x *= tf.math.sqrt(tf.cast(self.dim_x, tf.float32))
        x += self.pos_encoding[:tf.shape(x)[1],:]
        
        attn_weights = {}
        for i in range(self.num_layers):
            x, attn_w1, attn_w2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask)
            attn_weights[f'decoder_layer_{i}_self_att'] = attn_w1
            attn_weights[f'decoder_layer_{i}_decenc_att'] = attn_w2
        
        xhat = self.ffn(x,training=training)
        return xhat, attn_weights

class yDecoder(tf.keras.layers.Layer):
    def __init__(self, dim_y, Ty, key_dim, fc_dim, ffn_dim, num_layers = 1, num_heads = 2, dropout_rate=0.1, layernorm_eps=1e-6):
        super(yDecoder, self).__init__()
        self.dim_y = dim_y
        self.Ty = Ty
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(self.Ty, self.dim_y).get_positional_encoding()
        self.dec_layers = [DecoderLayer(output_dim=self.dim_y, key_dim=key_dim, fc_dim=fc_dim, num_heads=num_heads, dropout_rate=dropout_rate,layernorm_eps=layernorm_eps) for _ in range(self.num_layers)]
        ## finally feed-forward network
        self.ffn = tf.keras.Sequential([
                        Dense(ffn_dim, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(self.dim_y)
                    ])
                    
    def call(self, y, enc_output, look_ahead_mask, training = False):
        """
        Forward  pass for the Decoder - low freq variable
        Argv:
            y -- Tensor of shape (batch_size, Ty, dim_y)
            enc_output --  Tensor of shape(batch_size, Lx, dim_x)
            training -- Boolean, set to true to activate the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
        Returns:
            y -- Tensor of shape (batch_size, 1, dim_y)
            attention_weights - Dictionary of tensors containing all the attention weights
        """
        
        y *= tf.math.sqrt(tf.cast(self.dim_y, tf.float32))
        y += self.pos_encoding[:tf.shape(y)[1],:]
        attn_weights = {}
        for i in range(self.num_layers):
            y, attn_w1, attn_w2 = self.dec_layers[i](y, enc_output, training, look_ahead_mask)
            attn_weights[f'decoder_layer_{i}_self_att'] = attn_w1
            attn_weights[f'decoder_layer_{i}_decenc_att'] = attn_w2
        
        yhat = self.ffn(y[:,-1,:],training=training)
        return yhat, attn_weights
        
class Transformer(tf.keras.Model):
    def __init__(
        self,
        dim_x,
        dim_y,
        Lx,
        Tx,
        Ty,
        key_dim_enc,    # encoder key dim
        fc_dim_enc,     # encoder fully-connected layer dim
        key_dim_xdec,   # x-decoder key dim
        fc_dim_xdec,    # x-decoder fully-connected layer dim
        ffn_dim_x,      # final feed-forward network layer dim
        key_dim_ydec,   # y-decoder key dim
        fc_dim_ydec,    # y-decoder fully-connected layer dim
        ffn_dim_y,      # final feed-forward network layer dim
        num_layers = 1,
        num_heads = 4,
        freq_ratio = 3,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
        bidirectional_encoder = False
    ):
        super(Transformer, self).__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.Lx = Lx
        self.Tx = Tx
        self.Ty = Ty
        
        self.freq_ratio = freq_ratio
        self.bidirectional_encoder = bidirectional_encoder
        
        ## transformer encoder for x
        self.encoder = Encoder(dim_x = self.dim_x,
                               Lx = self.Lx,
                               key_dim = key_dim_enc,
                               fc_dim = fc_dim_enc,
                               num_layers = num_layers,
                               num_heads = num_heads,
                               dropout_rate = dropout_rate,
                               layernorm_eps = layernorm_eps)
        self.encoder_look_ahead_mask = self.create_look_ahead_mask(self.Lx) if not bidirectional_encoder else None
        ## transformer decoder for x
        self.xdecoder = xDecoder(dim_x = self.dim_x,
                                 Tx = self.Tx,
                                 key_dim = key_dim_xdec,
                                 fc_dim = fc_dim_xdec,
                                 ffn_dim = ffn_dim_x,
                                 num_layers = num_layers,
                                 num_heads = num_heads,
                                 dropout_rate = dropout_rate,
                                 layernorm_eps = layernorm_eps)
        ## transformer decoder for y
        self.ydecoder = yDecoder(dim_y = self.dim_y,
                                 Ty = self.Ty,
                                 key_dim = key_dim_ydec,
                                 fc_dim = fc_dim_ydec,
                                 ffn_dim = ffn_dim_y,
                                 num_layers = num_layers,
                                 num_heads = num_heads,
                                 dropout_rate = dropout_rate,
                                 layernorm_eps = layernorm_eps)
        
        
    ## helper function
    def create_look_ahead_mask(self, sequence_length):
        """
        Argv: sequence_length -- sequence length
        Return: mask -- (sequence_length, sequence_length) tensor; only lower triangular (incl. diagonal) are filled with ones, with 1 means to attend; see https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        """
        mask = tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
        return mask
    
    def call(self, batch_inputs, training=False):
    
        x_encoder_in, x_decoder_in, y_decoder_in = batch_inputs
        batch_size = tf.shape(x_encoder_in)[0]
        
        ## stage 1: transformer encoder for x, no look-ahead-mask
        enc_output = self.encoder(x_encoder_in, look_ahead_mask = self.encoder_look_ahead_mask, training = training)
        
        ## stage 2.1: transformer decoder for x
        look_ahead_mask_xdecoder = self.create_look_ahead_mask(x_decoder_in.shape[1])
        x_pred, attn_weights_xdec = self.xdecoder(x_decoder_in, enc_output, look_ahead_mask_xdecoder, training)
        
        ## stage 2.2: transformer decoder for y
        look_ahead_mask_ydecoder = self.create_look_ahead_mask(y_decoder_in.shape[1])
        y_pred, attn_weights_ydec = self.ydecoder(y_decoder_in, enc_output, look_ahead_mask_ydecoder, training)
        
        return [x_pred, y_pred]

    def build_graph(self):
        x_encoder_in = tf.keras.layers.Input(shape=(self.Lx,self.dim_x))
        x_decoder_in = tf.keras.layers.Input(shape=(self.Tx,self.dim_x))
        y_decoder_in = tf.keras.layers.Input(shape=(self.Ty,self.dim_y))
        inputs = [x_encoder_in, x_decoder_in, y_decoder_in]
        outputs = self.call(inputs)
        return Model(inputs=inputs, outputs=outputs)
        
class TransformerPred(_baseSeqPred):

    def __init__(
        self,
        model,
        scaler_x,
        scaler_y,
        apply_inv_scaler=True
    ):
        super().__init__(model, scaler_x, scaler_y, apply_inv_scaler)
