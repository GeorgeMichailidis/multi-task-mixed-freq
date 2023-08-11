"""
Forecasting using simple exponential smoothing
"""

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import pandas as pd

class SimpleExpSmoother():
    """
    simple exponential smoothing
    """
    def __init__(self,alpha=0.5):
        self.alpha = alpha
        self.optimized = True if alpha is None else False
        
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
        
