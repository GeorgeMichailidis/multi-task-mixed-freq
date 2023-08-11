import math
import numpy as np

class Evaluator():
    """
    evaluate the discrepancy between truth and pred, both are of shape (num_of_steps, dim) = (n, p)
    """
    def __init__(self, batched = False):
        self.batched = batched
    
    def batch_eval(self, method, truth, pred):
        batch_size = truth.shape[0]
        fn = getattr(self, method)
        metric_val = []
        for i in range(batch_size):
            metric_val.append(fn(truth[i], pred[i]))
        metric_val = np.stack(metric_val,axis=0)
        return np.mean(metric_val,axis=0)
    
    def mse(self, truth, pred):
        """
        float, ||truth - pred||^2_F/(n*p)
        """
        return np.mean((pred - truth)**2)
    
    def rmse(self, truth, pred):
        """
        float, sqrt of mse
        """
        return np.sqrt(self.mse(truth,pred))

    def rmse_by_step(self, truth, pred):
        """
        if not batched: array of size (n,); for each step i, sqrt(||truth[i] - pred[i]||^2_2/p)
        """
        return np.linalg.norm(pred-truth, axis=1)/np.sqrt(pred.shape[1])
        
    def mspe_by_step(self, truth, pred):
        """
        array of size (n,); for each step i, sum(|err[i,j]|/truth[i,j])^2/p
        """
        sq_pct_err = np.square(np.abs(pred-truth)/np.abs(truth))
        return np.mean(sq_pct_err,axis=1)

    def rmspe_by_step(self,truth,pred):
        """
        array of size (n,); for each step i, sqrt(mean_across_j(|err[i,j]|/truth[i,j])^2))
        """
        return np.sqrt(self.rmspe_by_step(self,truth,pred))
        
    def rel_l2err_by_step(self, truth, pred):
        """
        relative error in terms of L2
        array of size (n,)
        for each step i, ||truth[i] - pred[i]||/||truth[i]||
        """
        return np.linalg.norm(pred-truth,axis=1)/np.linalg.norm(truth,axis=1)

    def avg_mape_by_step(self, truth, pred):
        """
        averaged mape across coordinates, by step
        array of size (n,)
        for each step i, average_across_j(|truth[i,j] - pred[i,j]|/|truth[i,j]|)
        """
        return np.mean(np.abs(pred-truth)/np.abs(truth),axis=1)
    
    def median_mape_by_step(self, truth, pred):
        """
        median mape across coordinates, by step
        array of size (n,)
        for each step i, median_across_j(|truth[i,j] - pred[i,j]|/|truth[i,j]|)
        """
        return np.median(np.abs(pred-truth)/np.abs(truth),axis=1)
