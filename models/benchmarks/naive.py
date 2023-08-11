"""
Naive forecasting where the forecasted values are held flat at the last known one
Here we leverage the implementation of SimpleExpSmoothing with alpha=1
"""
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import pandas as pd

class Naive():
    """
    naive
    """
    def __init__(self,alpha=1):
    
        self.alpha = alpha
        self.optimized = False
        
    def forecast(self, x, horizon):
        if isinstance(x, pd.DataFrame): x = x.values
        p = x.shape[1]
        forecasts = []
        for j in range(p):
            fit = SimpleExpSmoothing(x[:,j]).fit(smoothing_level=self.alpha,optimized=self.optimized)
            fcast = fit.forecast(horizon)
            forecasts.append(fcast)
        forecasts = np.stack(forecasts,axis=1)
        return forecasts
        
