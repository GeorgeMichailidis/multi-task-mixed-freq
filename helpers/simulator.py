
import sys

import numpy as np
import pickle
from datetime import datetime

from .utils_data import *

class Simulator():
    def __init__(self):
        self.default_seed_map = {'ss00' : 1001,
                                 'ss01' : 2002,
                                 'ss02' : 3003,
                                 'regr01' : 4004,
                                 'regr02' : 5005,
                                }
        
    def generate_dataset(self, ds_id, n, replica_id = 0, save_to_excel = True, plot = True):
        d = {
                'ss00': ('GetDS_ss00','state-space, small system, linear dynamics'),
                'ss01': ('GetDS_ss01','state-space, medium system, mildly nonlinear dynamics'),
                'ss02': ('GetDS_ss02','state-space, large system, nonlinear dynamics'),
                'regr01': ('GetDS_regr01','regression, medium system, mildly nonlinear dynamics'),
                'regr02': ('GetDS_regr02','regression, large system, nonlinear dynamics')
            }
        
        cls_name, desc = d[ds_id]
        generator = globals()[cls_name]()
        print(desc)

        seed_id = self.default_seed_map[ds_id] + replica_id
        np.random.seed(seed_id)
        
        if 'ss' in ds_id:
            fdata, xdata, ydata = generator.gen_datasets(n, seed_id = seed_id, save_to_excel = save_to_excel, plot = plot)
            return fdata, xdata, ydata
        else:
            xdata, ydata = generator.gen_datasets(n, seed_id = seed_id, save_to_excel = save_to_excel, plot = plot)
            return xdata, ydata

class RBFNConverter():
    """
    radial basis function network converter
    """
    def __init__(self,input_dim=10,hidden_dim=100,output_dim=30,rescale=True,center_range=[-2,2],scale_range=[0.1,0.5]):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rescale = rescale
        self.center_range = center_range
        self.scale_range = scale_range
        ## rbf centers and scales
        self.centers = np.random.uniform(low=self.center_range[0],high=self.center_range[1],size=(self.hidden_dim,input_dim))
        self.scales = np.random.uniform(low=0.1,high=0.5,size=(self.hidden_dim,))
        ## weights to the output layer
        self.weights = np.random.uniform(low=-np.sqrt(6)/np.sqrt(self.hidden_dim+self.output_dim),
                                 high=np.sqrt(6)/np.sqrt(self.hidden_dim+self.output_dim),
                                 size=(self.output_dim,self.hidden_dim))
                                 
    def convert(self,x):
    
        if isinstance(x, list):
            x = np.concatenate(x).ravel()
        else:
            x = np.squeeze(x)
            
        x = np.expand_dims(x,axis=0) ## x.shape = (1, input_dim)
        hidden_neurons = np.exp(- self.scales * np.sum((self.centers - x)**2,axis=1))
        output_neurons = np.dot(self.weights,hidden_neurons)
        if self.rescale:
            output_neurons *= np.std(x)/np.std(output_neurons)
        return np.squeeze(output_neurons)

class StateSpaceCoefGenerator():
    """
    coefficient generator when the DGP is of state space type
    """
    def __init__(self,dim_f,nlags_ff,dim_x,nlags_xx,nlags_fx,dim_y,nlags_yy,nlags_fy):
        self.dim_f = dim_f
        self.nlags_ff = nlags_ff
        self.dim_x = dim_x
        self.nlags_xx = nlags_xx
        self.nlags_fx = nlags_fx
        self.dim_y = dim_y
        self.nlags_yy = nlags_yy
        self.nlags_fy = nlags_fy
        
    def gen_coef(self):
    
        ## for the f process
        if self.nlags_ff == 1:
            A_ff = gen_VAR1_coef(self.dim_f, target_sr = 0.5, scale_range=(1,2), diag = False)
            A_ff = A_ff[:,:,np.newaxis]
        else:
            A_ff = gen_VARd_coef(self.dim_f, d = self.nlags_ff, target_sr = 0.5, diag = False, scale_ranges = [(1,2) for _ in range(self.nlags_ff)])
        
        ## for the x process
        if self.nlags_xx is not None:
            if self.nlags_xx == 1:
                A_xx = gen_VAR1_coef(self.dim_x, target_sr = 0.5, scale_range=(1,2), diag = False)
                A_xx = A_xx[:,:,np.newaxis]
            else:
                A_xx = gen_VARd_coef(self.dim_x, d = self.nlags_xx, target_sr = 0.5, diag = False, scale_ranges=[(1,2) for _ in range(self.nlags_xx)])
        else:
            A_xx = None
        
        Lambdas_fx = gen_exp_almon_mtx(self.dim_x, self.dim_f, nlags = self.nlags_fx, normalize = False, deltas = None, degree = 2, low = -0.25, high = 0.25, sort = True, pm = True)
        
        ## for the y process
        if self.nlags_yy is not None:
            if self.nlags_yy == 1:
                A_yy = gen_VAR1_coef(self.dim_y, target_sr = 0.5, scale_range=(1,2), diag = False)
                A_yy = A_yy[:,:,np.newaxis]
            else:
                A_yy = gen_VARd_coef(self.dim_y, d = self.nlags_yy, target_sr = 0.5, diag = False, scale_ranges=[(1,2) for _ in range(self.nlags_yy)])
        else:
            A_yy = None
        
        Lambdas_fy = gen_exp_almon_mtx(self.dim_y, self.dim_f, nlags = self.nlags_fy, normalize = False, deltas = None, degree = 2, low = -0.25, high = 0.25, sort = True, pm = True)
        
        return A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy
        
class RegrCoefGenerator():
    """
    coefficient generator when the DGP is of regression type
    """
    def __init__(self,dim_x,nlags_xx,dim_y,nlags_yy,nlags_xy):
        self.dim_x = dim_x
        self.nlags_xx = nlags_xx
        self.dim_y = dim_y
        self.nlags_yy = nlags_yy
        self.nlags_xy = nlags_xy
        
    def gen_coef(self):
    
        ## for the x process
        if self.nlags_xx == 1:
            A_xx = gen_VAR1_coef(self.dim_x, target_sr = 0.5, scale_range=(1,2), diag = False)
            A_xx = A_xx[:,:,np.newaxis]
        else:
            A_xx = gen_VARd_coef(self.dim_x, d = self.nlags_xx, target_sr = 0.5, diag = False, scale_ranges=[(1,2) for _ in range(self.nlags_xx)])
        ## for the y process
        if self.nlags_yy == 1:
            A_yy = gen_VAR1_coef(self.dim_y, target_sr = 0.5, scale_range=(1,2), diag = False)
            A_yy = A_yy[:,:,np.newaxis]
        else:
            A_yy = gen_VARd_coef(self.dim_y, d = self.nlags_yy, target_sr = 0.5, diag = False, scale_ranges=[(1,2) for _ in range(self.nlags_yy)])
        Lambdas_xy = gen_exp_almon_mtx(self.dim_y, self.dim_x, nlags = self.nlags_xy, normalize = True, deltas = None, degree = 2, low = -0.25, high = 0.25, sort = True, pm = True)
        return A_xx, A_yy, Lambdas_xy

######################################
## state-space models start from here
######################################

class GetDS_ss00(StateSpaceCoefGenerator):
    def __init__(
        self,
        ## basic dimension and lag specifications
        dim_f = 5,
        nlags_ff = 2,
        dim_x = 50,
        nlags_xx = 1,
        nlags_fx = 6,
        dim_y = 10,
        nlags_yy = 1,
        nlags_fy = 15,
        ## the scale of the random noise
        noise_f = 1,
        noise_x = 1,
        noise_y = 1,
        ## burning
        BURN = 600
    ):
        super().__init__(dim_f,nlags_ff,dim_x,nlags_xx,nlags_fx,dim_y,nlags_yy,nlags_fy)
        self.noise_f = noise_f
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.BURN = BURN
        
    def make_one_dataset(self, n, A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy):
        
        SAMPLE_SIZE = n + self.BURN
        SAMPLE_SIZE_y = int(SAMPLE_SIZE/3)
        
        ## take care of f - linear autoregressive
        fdata = np.zeros((SAMPLE_SIZE,self.dim_f))
        fdata[:self.nlags_ff,:] = np.random.normal(loc=0,scale=self.noise_f,size=(self.nlags_ff,self.dim_f))
        for t in range(self.nlags_ff,SAMPLE_SIZE):
            signal = np.zeros((self.dim_f,))
            for k in range(1,self.nlags_ff + 1):
                signal += np.dot(A_ff[:,:,k-1],fdata[t-k,:].T)
            fdata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_f,size=(self.dim_f,))
        
        ## take care of x
        xdata = np.zeros((SAMPLE_SIZE,self.dim_x))
        xdata[:max(self.nlags_fx,self.nlags_xx),:] = np.random.standard_t(8,size=(max(self.nlags_fx,self.nlags_xx),self.dim_x)) * self.noise_x/np.sqrt(8/(8-2))
        for t in range(max(self.nlags_fx, self.nlags_xx),SAMPLE_SIZE):
            signal = np.zeros((self.dim_x,))
            for k in range(1,self.nlags_xx+1):
                signal += np.dot(A_xx[:,:,k-1],xdata[t-k,:].T)
            for k in range(self.nlags_fx):
                signal += np.dot(Lambdas_fx[:,:,k], fdata[t-k,:].T)
            xdata[t,:] = signal + np.random.standard_t(8,size=(self.dim_x,)) * self.noise_x/np.sqrt(8/(8-2))
            
        ## take care of y
        ydata = np.zeros((SAMPLE_SIZE_y,self.dim_y))
        y_valid_start = max(self.nlags_fy//3,self.nlags_yy)
        ydata[:y_valid_start,:] = np.random.normal(loc=0,scale=self.noise_y,size=(y_valid_start,self.dim_y))
        
        for t in range(y_valid_start,SAMPLE_SIZE_y):
            f_idx = 3*(t+1)-1 ## end of period index
            signal = np.zeros((self.dim_y,))
            for k in range(1,self.nlags_yy+1):
                signal += np.dot(A_yy[:,:,k-1],ydata[t-k,:].T)
            for k in range(self.nlags_fy):
                signal += np.dot(Lambdas_fy[:,:,k], fdata[f_idx-k,:].T)
            ydata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_y,size=(self.dim_y,))
                        
        fdata = fdata[self.BURN:,:]
        xdata = xdata[self.BURN:,:]
        ydata = ydata[int(self.BURN//3):,:]

        return fdata, xdata, ydata

    def gen_datasets(self, n, seed_id = 1, filename = None, save_to_excel = False, plot = True):
        
        A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy = self.gen_coef()
        fdata, xdata, ydata = self.make_one_dataset(n, A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy)
        
        if filename is None:
            filename = f'data_sim/ss00_{seed_id}'
        
        data = {'f':fdata,'x':xdata,'y':ydata}
        coef = {'A_ff': A_ff, 'A_xx': A_xx, 'Lambdas_fx': Lambdas_fx, 'A_yy': A_yy, 'Lambdas_fy': Lambdas_fy}
        meta_data = {'data': data, 'coef': coef}
        
        with open(f"{filename}.pickle","wb") as handle:
            pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        if save_to_excel: export_data_to_excel(data,filename)
        if plot: plot_data_rand_col(data,filename)
        
        return fdata, xdata, ydata
        
class GetDS_ss01(StateSpaceCoefGenerator):
    def __init__(
        self,
        ## basic dimension and lag specifications
        dim_f = 5,
        nlags_ff = 2,
        dim_x = 50,
        nlags_xx = 1,
        nlags_fx = 12,
        dim_y = 10,
        nlags_yy = 1,
        nlags_fy = 6,
        ## the scale of the random noise
        noise_f = 1,
        noise_x = 1,
        noise_y = 1,
        ## burning
        BURN = 600
    ):
        super().__init__(dim_f,nlags_ff,dim_x,nlags_xx,nlags_fx,dim_y,nlags_yy,nlags_fy)
        self.noise_f = noise_f
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.converter_fx = RBFNConverter(input_dim=self.dim_f,hidden_dim=50,output_dim=self.dim_f)
        self.converter_fy = RBFNConverter(input_dim=self.dim_f,hidden_dim=20,output_dim=self.dim_f)
        self.BURN = BURN
        
    def make_one_dataset(self, n, A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy):
        
        SAMPLE_SIZE = n + self.BURN
        SAMPLE_SIZE_y = int(SAMPLE_SIZE/3)
        
        ## take care of f
        fdata = np.zeros((SAMPLE_SIZE,self.dim_f))
        fdata[:self.nlags_ff,:] = np.random.normal(loc=0,scale=self.noise_f,size=(self.nlags_ff,self.dim_f))
        for t in range(self.nlags_ff,SAMPLE_SIZE):
            signal = np.zeros((self.dim_f,))
            for k in range(1,self.nlags_ff + 1):
                signal += np.dot(A_ff[:,:,k-1],fdata[t-k,:].T)
            fdata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_f,size=(self.dim_f,))
            
        ## take care of x
        xdata = np.zeros((SAMPLE_SIZE,self.dim_x))
        xdata[:max(self.nlags_fx,self.nlags_xx),:] = np.random.standard_t(8,size=(max(self.nlags_fx,self.nlags_xx),self.dim_x)) * self.noise_x/np.sqrt(8/(8-2))
        for t in range(max(self.nlags_fx, self.nlags_xx),SAMPLE_SIZE):
            signal = np.zeros((self.dim_x,))
            ## x autoregressive, linear
            for k in range(1,self.nlags_xx+1):
                signal += np.dot(A_xx[:,:,k-1],xdata[t-k,:].T)
            ## f -> x, partially non-linear
            for k in range(self.nlags_fx):
                if k <= self.nlags_fx//2:
                    signal += np.dot(Lambdas_fx[:,:,k], fdata[t-k,:].T)
                else:
                    signal += np.dot(Lambdas_fx[:,:,k], self.converter_fx.convert(fdata[t-k,:]))
            xdata[t,:] = signal + np.random.standard_t(8,size=(self.dim_x,)) * self.noise_x/np.sqrt(8/(8-2))
                
        ## take care of y
        ydata = np.zeros((SAMPLE_SIZE_y,self.dim_y))
        y_valid_start = max(self.nlags_fy//3,self.nlags_yy)
        ydata[:y_valid_start,:] = np.random.normal(loc=0,scale=self.noise_y,size=(y_valid_start,self.dim_y))
        for t in range(y_valid_start,SAMPLE_SIZE_y):
            f_idx = 3*(t+1)-1 ## end of period index
            signal = np.zeros((self.dim_y,))
            ## y autoregressive, linear
            for k in range(1,self.nlags_yy+1):
                signal += np.dot(A_yy[:,:,k-1],ydata[t-k,:].T)
            ## f -> y, partially non-linear
            for k in range(self.nlags_fy):
                if k <= self.nlags_fy//2:
                    signal += np.dot(Lambdas_fy[:,:,k], fdata[f_idx-k,:].T)
                else:
                    signal += np.dot(Lambdas_fy[:,:,k], self.converter_fy.convert(fdata[f_idx-k,:]))
            ydata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_y,size=(self.dim_y,))

        fdata = fdata[self.BURN:,:]
        xdata = xdata[self.BURN:,:]
        ydata = ydata[int(self.BURN//3):,:]
        
        return fdata, xdata, ydata

    def gen_datasets(self, n, seed_id, filename = None, save_to_excel = False, plot = False):

        A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy = self.gen_coef()
        fdata, xdata, ydata = self.make_one_dataset(n, A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy)
        
        if filename is None:
            filename = f'data_sim/ss01_{seed_id}'
        
        data = {'f':fdata,'x':xdata,'y':ydata}
        coef = {'A_ff': A_ff, 'A_xx': A_xx, 'Lambdas_fx': Lambdas_fx, 'A_yy': A_yy, 'Lambdas_fy': Lambdas_fy}
        meta_data = {'data': data, 'coef': coef}
        
        with open(f"{filename}.pickle","wb") as handle:
            pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        if save_to_excel: export_data_to_excel(data,filename)
        if plot: plot_data_rand_col(data,filename)
        
        return fdata, xdata, ydata

class GetDS_ss02(StateSpaceCoefGenerator):
    def __init__(
        self,
        ## basic dimension and lag specifications
        dim_f = 5,
        nlags_ff = 2,
        dim_x = 100,
        nlags_xx = 1,
        nlags_fx = 6,
        dim_y = 10,
        nlags_yy = 1,
        nlags_fy = 6,
        ## the scale of the random noise
        noise_f = 1,
        noise_x = 1,
        noise_y = 1,
        ## burning
        BURN = 600
    ):
        super().__init__(dim_f,nlags_ff,dim_x,nlags_xx,nlags_fx,dim_y,nlags_yy,nlags_fy)
        self.noise_f = noise_f
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.converter_fx = RBFNConverter(input_dim=self.dim_f,hidden_dim=50,output_dim=self.dim_f)
        self.converter_fy = RBFNConverter(input_dim=self.dim_f,hidden_dim=20,output_dim=self.dim_f)
        self.BURN = BURN

    def make_one_dataset(self, n, A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy):
        
        SAMPLE_SIZE = n + self.BURN
        SAMPLE_SIZE_y = int(SAMPLE_SIZE/3)
        
        ## take care of f
        fdata = np.zeros((SAMPLE_SIZE,self.dim_f))
        fdata[:self.nlags_ff,:] = np.random.normal(loc=0,scale=self.noise_f,size=(self.nlags_ff,self.dim_f))
        for t in range(self.nlags_ff,SAMPLE_SIZE):
            signal = np.zeros((self.dim_f,))
            for k in range(1,self.nlags_ff + 1):
                signal += np.dot(A_ff[:,:,k-1],fdata[t-k,:].T)
            fdata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_f,size=(self.dim_f,))
        
        ## take care of x
        xdata = np.zeros((SAMPLE_SIZE,self.dim_x))
        xdata[:max(self.nlags_fx,self.nlags_xx),:] = np.random.standard_t(8,size=(max(self.nlags_fx,self.nlags_xx),self.dim_x)) * self.noise_x/np.sqrt(8/(8-2))
        for t in range(max(self.nlags_fx, self.nlags_xx),SAMPLE_SIZE):
            signal = np.zeros((self.dim_x,))
            ## x autoregressive, linear
            for k in range(1,self.nlags_xx+1):
                signal += np.dot(A_xx[:,:,k-1],xdata[t-k,:].T)
            ## f -> x, non-linear
            for k in range(self.nlags_fx):
                if k % 2 == 0:
                    signal += np.dot(Lambdas_fx[:,:,k], self.converter_fx.convert(fdata[t-k,:]))
                else:
                    signal += np.dot(Lambdas_fx[:,:,k], self.converter_fx.convert(fdata[t-k+1,:]+fdata[t-k,:]))
            xdata[t,:] = signal + np.random.standard_t(8,size=(self.dim_x,)) * self.noise_x/np.sqrt(8/(8-2))
                
        ## take care of y
        ydata = np.zeros((SAMPLE_SIZE_y,self.dim_y))
        y_valid_start = max(self.nlags_fy//3,self.nlags_yy)
        ydata[:y_valid_start,:] = np.random.normal(loc=0,scale=self.noise_y,size=(y_valid_start,self.dim_y))
        for t in range(y_valid_start,SAMPLE_SIZE_y):
            f_idx = 3*(t+1)-1 ## end of period index
            signal = np.zeros((self.dim_y,))
            ## y autoregressive, linear
            for k in range(1,self.nlags_yy+1):
                signal += np.dot(A_yy[:,:,k-1],ydata[t-k,:].T)
            ## f -> y, non-linear
            for k in range(self.nlags_fy):
                if k % 2 == 0:
                    signal += np.dot(Lambdas_fy[:,:,k], self.converter_fy.convert(fdata[f_idx-k,:]))
                else:
                    signal += np.dot(Lambdas_fy[:,:,k], self.converter_fy.convert(fdata[f_idx-k+1,:]/2+fdata[f_idx-k,:]/2))
            ydata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_y,size=(self.dim_y,))

        fdata = fdata[self.BURN:,:]
        xdata = xdata[self.BURN:,:]
        ydata = ydata[int(self.BURN//3):,:]
        
        return fdata, xdata, ydata

    def gen_datasets(self, n, seed_id = 1, filename = None, save_to_excel = False, plot = False):
        
        A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy = self.gen_coef()
        fdata, xdata, ydata = self.make_one_dataset(n, A_ff, A_xx, Lambdas_fx, A_yy, Lambdas_fy)
        
        if filename is None:
            filename = f'data_sim/ss02_{seed_id}'

        data = {'f':fdata,'x':xdata,'y':ydata}
        coef = {'A_ff': A_ff, 'A_xx': A_xx, 'Lambdas_fx': Lambdas_fx, 'A_yy': A_yy, 'Lambdas_fy': Lambdas_fy}
        meta_data = {'data': data, 'coef': coef}
        
        with open(f"{filename}.pickle","wb") as handle:
            pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

        if save_to_excel: export_data_to_excel(data,filename)
        if plot: plot_data_rand_col(data,filename)
        
        return fdata, xdata, ydata

#####################################
## regression models
#####################################

class GetDS_regr01(RegrCoefGenerator):
    def __init__(
        self,
        ## basic dimension and lag specifications
        dim_x = 50,
        nlags_xx = 2,
        dim_y = 10,
        nlags_yy = 1,
        nlags_xy = 6,
        ## the scale of the random noise
        noise_x = 2,
        noise_y = 1,
        ## burning
        BURN = 600
    ):
        super().__init__(dim_x,nlags_xx,dim_y,nlags_yy,nlags_xy)
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.converter_xy = RBFNConverter(input_dim=self.dim_x,hidden_dim=20,output_dim=self.dim_x)
        self.BURN = BURN

    def make_one_dataset(self, n, A_xx, A_yy, Lambdas_xy):
        
        SAMPLE_SIZE = n + self.BURN
        SAMPLE_SIZE_y = int(SAMPLE_SIZE/3)
        
        ## take care of x
        xdata = np.zeros((SAMPLE_SIZE,self.dim_x))
        xdata[:self.nlags_xx,:] = np.random.standard_t(8,size=(self.nlags_xx,self.dim_x)) * self.noise_x/np.sqrt(8/(8-2))
        for t in range(self.nlags_xx,SAMPLE_SIZE):
            signal = np.zeros((self.dim_x,))
            ## x autoregressive, linear dynamic apart from non-linear dependency on the last lag: Xt = A_1X_{t-1} + A_2(X_{t-2} * sin(X_{t-2})) + noise
            for k in range(1,self.nlags_xx):
                signal += np.dot(A_xx[:,:,k-1],xdata[t-k,:].T)
            signal += np.dot(A_xx[:,:,self.nlags_xx-1], (xdata[t-self.nlags_xx,:]*np.sin(xdata[t-self.nlags_xx,:])).T)
            xdata[t,:] = signal + np.random.standard_t(8,size=(self.dim_x,)) * self.noise_x/np.sqrt(8/(8-2))
                
        ## take care of y
        ydata = np.zeros((SAMPLE_SIZE_y,self.dim_y))
        y_valid_start = max(self.nlags_xy//3,self.nlags_yy)
        ydata[:y_valid_start,:] = np.random.normal(loc=0,scale=self.noise_y,size=(y_valid_start,self.dim_y))
        for t in range(y_valid_start,SAMPLE_SIZE_y):
            x_idx = 3*(t+1)-1 ## end of period index
            signal = np.zeros((self.dim_y,))
            ## y autoregressive, linear autoregressive dynamic
            for k in range(1,self.nlags_yy+1):
                signal += np.dot(A_yy[:,:,k-1],ydata[t-k,:].T)
            ## x -> y, partially nonlinear
            for k in range(self.nlags_xy):
                if k <= self.nlags_xy//2:
                    signal += np.dot(Lambdas_xy[:,:,k], xdata[x_idx-k,:].T)
                else:
                    signal += np.dot(Lambdas_xy[:,:,k], self.converter_xy.convert(xdata[x_idx-k,:]).T)
            ydata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_y,size=(self.dim_y,))

        xdata = xdata[self.BURN:,:]
        ydata = ydata[int(self.BURN//3):,:]
        
        return xdata, ydata

    def gen_datasets(self, n, seed_id = 1, filename = None, save_to_excel = False, plot = False):

        A_xx, A_yy, Lambdas_xy = self.gen_coef()
        xdata, ydata = self.make_one_dataset(n, A_xx, A_yy, Lambdas_xy)
        
        if filename is None:
            filename = f'data_sim/regr01_{seed_id}'
        
        data = {'x':xdata,'y':ydata}
        coef = {'A_xx': A_xx, 'A_yy': A_yy, 'Lambdas_xy': Lambdas_xy}
        meta_data = {'data': data, 'coef': coef}
        
        with open(f"{filename}.pickle","wb") as handle:
            pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        if save_to_excel: export_data_to_excel(data,filename)
        if plot: plot_data_rand_col(data,filename)
            
        return xdata, ydata

class GetDS_regr02(RegrCoefGenerator):
    def __init__(
        self,
        ## basic dimension and lag specifications
        dim_x = 100,
        nlags_xx = 2,
        dim_y = 20,
        nlags_yy = 1,
        nlags_xy = 6,
        ## the scale of the random noise
        noise_x = 2,
        noise_y = 1,
        ## burning
        BURN = 600
    ):
        super().__init__(dim_x,nlags_xx,dim_y,nlags_yy,nlags_xy)
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.converter_xy = RBFNConverter(input_dim=self.dim_x,hidden_dim=20,output_dim=self.dim_x)
        self.BURN = BURN

    def make_one_dataset(self, n, A_xx, A_yy, Lambdas_xy):
        
        SAMPLE_SIZE = n + self.BURN
        SAMPLE_SIZE_y = int(SAMPLE_SIZE/3)
        
        ## take care of x
        xdata = np.zeros((SAMPLE_SIZE,self.dim_x))
        xdata[:self.nlags_xx,:] = np.random.standard_t(8,size=(self.nlags_xx,self.dim_x)) * self.noise_x/np.sqrt(8/(8-2))
        for t in range(self.nlags_xx,SAMPLE_SIZE):
            signal = np.zeros((self.dim_x,))
            ## x autoregressive, linear dynamic apart from the last lag: Xt = A_1X_{t-1} + A_2 (X_{t-2} * sin(X_{t-2})) + noise
            for k in range(1,self.nlags_xx):
                signal += np.dot(A_xx[:,:,k-1],xdata[t-k,:].T)
            signal += np.dot(A_xx[:,:,self.nlags_xx-1], (xdata[t-self.nlags_xx,:]*np.sin(xdata[t-self.nlags_xx,:])).T)
            xdata[t,:] = signal + np.random.standard_t(8,size=(self.dim_x,)) * self.noise_x/np.sqrt(8/(8-2))
                
        ## take care of y
        ydata = np.zeros((SAMPLE_SIZE_y,self.dim_y))
        y_valid_start = max(self.nlags_xy//3,self.nlags_yy)
        ydata[:y_valid_start,:] = np.random.normal(loc=0,scale=self.noise_y,size=(y_valid_start,self.dim_y))
        for t in range(y_valid_start,SAMPLE_SIZE_y):
            x_idx = 3*(t+1)-1 ## end of period index
            signal = np.zeros((self.dim_y,))
            ## y linear autoregressive
            for k in range(1,self.nlags_yy+1):
                signal += np.dot(A_yy[:,:,k-1],ydata[t-k,:].T)
            ## x -> y
            for k in range(self.nlags_xy):
                if k <= self.nlags_xy//2:
                    signal += np.dot(Lambdas_xy[:,:,k], self.converter_xy.convert(xdata[x_idx-k,:]).T)
                else:
                    signal += np.dot(Lambdas_xy[:,:,k], self.converter_xy.convert(xdata[x_idx-k+1,:]/2 + xdata[x_idx-k,:]/2).T)
            ydata[t,:] = signal + np.random.normal(loc=0,scale=self.noise_y,size=(self.dim_y,))
        
        xdata = xdata[self.BURN:,:]
        ydata = ydata[int(self.BURN//3):,:]
        return xdata, ydata

    def gen_datasets(self, n, seed_id = 1, filename = None, save_to_excel = False, plot = False):
    
        A_xx, A_yy, Lambdas_xy = self.gen_coef()
        xdata, ydata = self.make_one_dataset(n, A_xx, A_yy, Lambdas_xy)

        if filename is None: filename = f'data_sim/regr02_{seed_id}'

        data = {'x':xdata,'y':ydata}
        coef = {'A_xx': A_xx, 'A_yy': A_yy, 'Lambdas_xy': Lambdas_xy}
        meta_data = {'data': data, 'coef': coef}
        
        with open(f"{filename}.pickle","wb") as handle:
            pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

        if save_to_excel: export_data_to_excel(data,filename)
        plot_data_rand_col(data,filename)
        
        return xdata, ydata
