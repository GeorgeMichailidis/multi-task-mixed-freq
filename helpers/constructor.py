"""
Centralized model class instantiation based on args
"""

from models import MTMFSeq2Seq, MTMFSeq2SeqPred
from models import Transformer, TransformerPred
from models import MTMFSeq2One, MTMFSeq2OnePred
from models import TwoMLP, MLPPred
from models import TwoGBM, GBMPred
from helpers import _baseMFDP, MFDPOneStep, MFDPMultiStep
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ClsConstructor():

    def __init__(self, args):
        self.args = args
        self.supported_models = ['MTMFSeq2Seq',
                                 'transformer',
                                 'MTMFSeq2One',
                                 'MLP',
                                 'GBM',
                                 'DeepAR',
                                 'NHiTS',
                                 'ARIMA']
        assert args.model_type in self.supported_models
        
        
    def create_data_processor(self):
        
        args = self.args
        if args.scaler_type == 'standard':
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
        elif args.scaler_type == 'minmax':
            scaler_x, scaler_y = MinMaxScaler((-1,1)), MinMaxScaler((-1,1))
        else:
            raise ValueError('invalid scaler_type')
    
        if args.model_type in ['DeepAR', 'NHiTS']:
            dp = _baseMFDP(Lx = args.Lx,
                           Tx = args.Tx,
                           Ly = args.Ly,
                           Ty = args.Ty,
                           freq_ratio = args.freq_ratio,
                           scaler_x = None,
                           scaler_y = None)
        elif args.model_type in ['MTMFSeq2Seq','transformer']:
            dp = MFDPMultiStep(Lx = args.Lx,
                               Tx = args.Tx,
                               Ly = args.Ty-1,
                               Ty = 1,
                               freq_ratio = args.freq_ratio,
                               scaler_x = scaler_x,
                               scaler_y = scaler_y,
                               zero_pad = args.zero_pad)
        else:
            dp = MFDPOneStep(Lx = args.Lx,
                             Tx = args.Tx,
                             Ly = args.Ty - 1,
                             Ty = 1,
                             freq_ratio = args.freq_ratio,
                             scaler_x = scaler_x,
                             scaler_y = scaler_y,
                             zero_pad = args.zero_pad)
        return dp
        
    def create_model(self):
    
        args = self.args
        
        if args.model_type == 'MTMFSeq2Seq':
            model = MTMFSeq2Seq(dim_x = args.dim_x,
                                dim_y = args.dim_y,
                                Lx = args.Lx,
                                Tx = args.Tx,
                                Ty = args.Ty,
                                n_a = args.n_a,
                                n_s = args.n_s,
                                n_align_x = args.n_align_x,
                                n_align_y = args.n_align_y,
                                fc_x = args.fc_x,
                                fc_y = args.fc_y,
                                dropout_rate = args.dropout_rate,
                                freq_ratio = args.freq_ratio,
                                bidirectional_encoder = args.bidirectional_encoder,
                                l1reg = args.l1reg,
                                l2reg = args.l2reg)
        elif args.model_type == 'transformer':
            model = Transformer(args.dim_x,
                            args.dim_y,
                            args.Lx,
                            args.Tx,
                            args.Ty,
                            key_dim_enc = args.key_dim_enc,
                            fc_dim_enc = args.fc_dim_enc,
                            key_dim_xdec = args.key_dim_xdec,
                            fc_dim_xdec = args.fc_dim_xdec,
                            ffn_dim_x = args.ffn_dim_x,
                            key_dim_ydec = args.key_dim_ydec,
                            fc_dim_ydec = args.fc_dim_ydec,
                            ffn_dim_y = args.ffn_dim_y,
                            num_layers = args.num_layers,
                            num_heads = args.num_heads,
                            freq_ratio = args.freq_ratio,
                            dropout_rate = args.dropout_rate,
                            layernorm_eps=1e-6,
                            bidirectional_encoder = args.bidirectional_encoder)
        elif args.model_type == 'MTMFSeq2One':
            model = MTMFSeq2One(Lx = args.Lx,
                                dim_x = args.dim_x,
                                Ty = args.Ty,
                                dim_y = args.dim_y,
                                n_a = args.n_a,
                                n_s = args.n_s,
                                n_align = args.n_align,
                                fc_x = args.fc_x,
                                fc_y = args.fc_y,
                                dropout_rate = args.dropout_rate,
                                freq_ratio = args.freq_ratio,
                                bidirectional_encoder = args.bidirectional_encoder,
                                l1reg = args.l1reg,
                                l2reg = args.l2reg)
        elif args.model_type == 'MLP':
            model = TwoMLP(dim_x = args.dim_x,
                           dim_y = args.dim_y,
                           Lx = args.Lx,
                           Ly = args.Ty - 1,
                           hidden_dim_x = args.hidden_dim_x,
                           activation_x = args.activation_x,
                           architect_x = args.architect_x,
                           hidden_dim_y = args.hidden_dim_y,
                           activation_y = args.activation_y,
                           architect_y = args.architect_y,
                           dropout_rate = args.dropout_rate)
        elif args.model_type == 'GBM':
            model = TwoGBM(dim_x = args.dim_x,
                           dim_y = args.dim_y,
                           hyper_params_x = args.hyper_params_x,
                           hyper_params_y = args.hyper_params_y)
        
        return model

    def create_predictor(self, model, dp, apply_inv_scaler = True):
    
        args = self.args
        
        if args.model_type == 'MTMFSeq2Seq':
            predictor = MTMFSeq2SeqPred(model, dp.scaler_x, dp.scaler_y, apply_inv_scaler=apply_inv_scaler)
        elif args.model_type == 'transformer':
            predictor = TransformerPred(model, dp.scaler_x, dp.scaler_y, apply_inv_scaler=apply_inv_scaler)
        elif args.model_type  == 'MTMFSeq2One':
            predictor = MTMFSeq2OnePred(model, dp.scaler_x, dp.scaler_y, apply_inv_scaler=apply_inv_scaler)
        elif args.model_type == 'MLP':
            predictor = MLPPred(model, args.freq_ratio, dp.scaler_x, dp.scaler_y, apply_inv_scaler=apply_inv_scaler)
        elif args.model_type == 'GBM':
            predictor = GBMPred(model, args.freq_ratio, dp.scaler_x, dp.scaler_y, apply_inv_scaler=apply_inv_scaler)
        else:
            raise ValueError('invalid args.model_type')
        
        return predictor
