"""
utility functions for additional recording during training
"""
import numpy as np
import pandas as pd
import math
from datetime import datetime

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.gridspec import GridSpec

def plot_loss_over_epoch(history, args, save_as_file=""):

    fig = plt.figure(figsize=(10,4),constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax0 = fig.add_subplot(gs[:,0])
    ax0.plot(history.history['loss'], label='train',color='red')
    if history.history.get('val_loss') is not None:
        ax0.plot(history.history['val_loss'], label='validation',color='green')
    ax0.legend(loc='upper right')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(history.history['output_1_loss'], label='train',color='red')
    if history.history.get('val_output_1_loss') is not None:
        ax1.plot(history.history['val_output_1_loss'], label='validation',color='green')
    ax1.set_title('output_1: x')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(gs[1,1])
    ax2.plot(history.history['output_2_loss'], label='train',color='red')
    if history.history.get('val_output_2_loss') is not None:
        ax2.plot(history.history['val_output_2_loss'], label='validation',color='green')
    ax2.set_title('output_2: y')
    ax2.legend(loc='upper right')
    
    if len(save_as_file) > 0:
        fig.savefig(save_as_file)

def plot_fitted_val(args, pred, truth, x_col = None, y_col = None, time_steps = 100, save_as_file = True):
    
    x_pred, y_pred = pred
    x_truth, y_truth = truth
    
    if x_col is None:
        x_col = np.random.choice(list(range(args.dim_x)))
    if y_col is None:
        y_col = np.random.choice(list(range(args.dim_y)))
    
    if time_steps is not None: ## subset
        ticks = np.arange(time_steps)
        sample_ids = np.random.choice(a=list(range(x_pred.shape[0])),size=time_steps)
        x_pred, x_truth = x_pred[sample_ids], x_truth[sample_ids]
        y_pred, y_truth = y_pred[sample_ids], y_truth[sample_ids]
    else:
        ticks = np.arange(x_pred.shape[0])
    
    if args.Tx > 1:
        num_cols = math.ceil((args.Tx+1)/2.)
        fig, axs = plt.subplots(2, num_cols, figsize=(12,6), constrained_layout=True)
        for step_id in range(args.Tx):
            r, c = step_id//num_cols, step_id%num_cols
            axs[r,c].plot(ticks,x_pred[:, step_id, x_col],label='model',color='red')
            axs[r,c].plot(ticks,x_truth[:, step_id, x_col],label='truth',color='black')
            axs[r,c].legend(loc='lower right')
            axs[r,c].set_title(f'step_id={step_id+1}, x_col={x_col}, model vs. truth')
        axs[-1,-1].plot(ticks,y_pred[:,y_col],label='model',color='red')
        axs[-1,-1].plot(ticks,y_truth[:,y_col],label='truth',color='black')
        axs[-1,-1].legend()
        axs[-1,-1].set_title(f'y_col={y_col}, model vs. truth')
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12,3), constrained_layout=True)
        axs[0].plot(ticks,x_pred[:,x_col],label='model',color='red')
        axs[0].plot(ticks,x_truth[:,x_col],label='truth',color='black')
        axs[0].legend(loc='lower right')
        axs[0].set_title(f'step_id=1, x_col={x_col}, model vs. truth')
        axs[1].plot(ticks,y_pred[:,y_col],label='model',color='red')
        axs[1].plot(ticks,y_truth[:,y_col],label='truth',color='black')
        axs[1].legend()
        axs[1].set_title(f'y_col={y_col}, model vs. truth')
    if len(save_as_file) > 0:
        fig.savefig(save_as_file)
    plt.close()
    
def export_error_to_excel(df_x_err,df_y_err, x_err_collect, y_err_collect, output_folder):
    
    print('######################################')
    print('## rmse summary for x (step_4):')
    for tag in x_err_collect.keys():
        median = df_x_err.loc[(df_x_err.index=='step_4') & (df_x_err['metric'] == 'median'), tag].values[0]
        mean = df_x_err.loc[(df_x_err.index=='step_4') & (df_x_err['metric'] == 'mean'), tag].values[0]
        std = df_x_err.loc[(df_x_err.index=='step_4') & (df_x_err['metric'] == 'std'), tag].values[0]
        print(f'#    tag = {tag}; median = {median:.2f}, mean = {mean:.2f}, std = {std:.3f}')
    print('## rmse summary for y (step_1):')
    for tag in y_err_collect.keys():
        median = df_y_err.loc[(df_y_err.index=='step_1') & (df_y_err['metric'] == 'median'), tag].values[0]
        mean = df_y_err.loc[(df_y_err.index=='step_1') & (df_y_err['metric'] == 'mean'), tag].values[0]
        std = df_y_err.loc[(df_y_err.index=='step_1') & (df_y_err['metric'] == 'std'), tag].values[0]
        print(f'#    tag = {tag}; median = {median:.2f}, mean = {mean:.2f}, std = {std:.3f}')
    print('######################################')
    
    print(f'[exporting error summary to excel] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    with pd.ExcelWriter(f'{output_folder}/forecast_err.xlsx') as writer:
        df_x_err.to_excel(writer,sheet_name=f'summary_x_err',index=True)
        df_y_err.to_excel(writer,sheet_name=f'summary_y_err',index=True)
        
        for experiment_tag, arr in x_err_collect.items():
            df = pd.DataFrame(data=arr,columns=[f'step{i+1}' for i in range(arr.shape[1])],index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_excel(writer,sheet_name=f'x_{experiment_tag}',index=True)
        for experiment_tag, arr in x_err_collect.items():
            df = pd.DataFrame(data=arr,columns=[f'step{i+1}' for i in range(arr.shape[1])],index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_excel(writer,sheet_name=f'y_{experiment_tag}',index=True)
