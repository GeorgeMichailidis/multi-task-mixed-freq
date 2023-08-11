"""
python clean_fred.py --config='../configs/FRED/data_preproc/202207.yaml'
"""
import sys
import yaml
import argparse
import os
import numpy as np
import warnings

import pandas as pd
import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument('--config',default='../configs/FRED/data_preproc/202207.yaml')
parser.add_argument('--output_override', default='')

def main():

    """ main function for running simulation and record evaluation metrics"""
    global args
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
            
    if not hasattr(args,'MFILL'):
        setattr(args,'MFILL', {})
    if not hasattr(args,'QFILL'):
        setattr(args,'QFILL', {})
    
    if len(args.output_override):
        setattr(args, 'output_file', args.output_override)
    else:
        setattr(args, 'output_file', f'{args.folder}/{args.vintage}.xlsx')
    print(f'cleaned file will be saved to: {args.output_file}')
        
    sys.stdout = open(args.log_file, 'w')
        
    data_to_write, first_data_date = {}, []
    
    #####
    ## 0. read in data dictionary
    #####
    data_dict = pd.read_excel(f'{args.folder}/{args.dictionary_file}', sheet_name=args.dictionary_sheet)
    
    global variable_dict, trans_map
    variable_dict = dict(zip(data_dict['MNEMONIC'],data_dict['DESCRIPTION']))
    trans_map = dict(zip(data_dict['MNEMONIC'],data_dict['TRANSF_CODE']))
    
    mVar = data_dict.loc[ (data_dict['LARGE']==1) & (data_dict['FREQ']=='M'),'MNEMONIC'].to_list()
    qVar = data_dict.loc[ (data_dict['LARGE']==1) & (data_dict['FREQ']=='Q'),'MNEMONIC'].to_list()
    
    #####
    ## 1. process quarterly data
    #####
    QSTART, QEND = dt.datetime.strptime(args.QSTART,'%m/%d/%Y'), dt.datetime.strptime(args.QEND,'%m/%d/%Y')
    
    qd_lvls_raw, qd_lvls, qd_transf = process_quarterly(qVar, QSTART, QEND, first_data_date)
    data_to_write['QVAR_LEVELS'] = qd_lvls_raw
    data_to_write['QVAR_LEVELS_FILLED'] = qd_lvls
    data_to_write['y'] = qd_transf
    
    #####
    ## 2. process monthly data
    #####
    MSTART, MEND = dt.datetime.strptime(args.MSTART,'%m/%d/%Y'), dt.datetime.strptime(args.MEND,'%m/%d/%Y')
    
    md_lvls_raw, md_lvls, md_transf = process_monthly(mVar, MSTART, MEND, first_data_date)
    data_to_write['MVAR_LEVELS'] = md_lvls_raw
    data_to_write['MVAR_LEVELS_FILLED'] = md_lvls
    data_to_write['x'] = md_transf
    
    #####
    ## 3. handle small set
    #####
    mVar_small = data_dict.loc[ (data_dict['SMALL']==1) & (data_dict['FREQ']=='M'),'MNEMONIC'].to_list()
    qVar_small = data_dict.loc[ (data_dict['SMALL']==1) & (data_dict['FREQ']=='Q'),'MNEMONIC'].to_list()
    
    if args.nested:
        data_to_write['y_small'] = qd_transf.filter(items=qVar_small)
        data_to_write['x_small'] = md_transf.filter(items=mVar_small)
    else:
        ## process quarterly for the small set
        QSTART_s, QEND_s = dt.datetime.strptime(args.QSTART_small,'%m/%d/%Y'), dt.datetime.strptime(args.QEND_small,'%m/%d/%Y')
        qd_lvls_raw_s, qd_lvls_s, qd_transf_s = process_quarterly(qVar_small, QSTART_s, QEND_s, first_data_date = None)
        data_to_write['QVAR_LEVELS_S'] = qd_lvls_raw_s
        data_to_write['QVAR_LEVELS_FILLED_S'] = qd_lvls_s
        data_to_write['y_small'] = qd_transf_s
        
        MSTART_s, MEND_s = dt.datetime.strptime(args.MSTART_small,'%m/%d/%Y'), dt.datetime.strptime(args.MEND_small,'%m/%d/%Y')
        md_lvls_raw_s, md_lvls_s, md_transf_s = process_monthly(mVar_small, MSTART_s, MEND_s, first_data_date = None)
        data_to_write['MVAR_LEVELS_S'] = md_lvls_raw_s
        data_to_write['MVAR_LEVELS_FILLED_S'] = md_lvls_s
        data_to_write['x_small'] = md_transf_s
    
    missing_qvar = list(set(qVar_small) - set(list(data_to_write['y_small'].columns)))
    if missing_qvar:
        warnings.warning(f'the following quarterly variabes are missing from the small set: {missing_qvar}')

    missing_mvar = list(set(mVar_small) - set(list(data_to_write['x_small'].columns)))
    if missing_mvar:
        warnings.warning(f'the following monthly variabes are missing from the small set: {missing_mvar}')
       
    #####
    ## 4. write
    #####
    first_data_date = pd.DataFrame(first_data_date)
    print(f'writing file to {args.output_file} ...')
    with pd.ExcelWriter(args.output_file) as writer:
        for key, df in data_to_write.items():
            print(f" >> sheet_name = {key}, shape = {df.shape}, from = {df.index[0].strftime('%Y-%m-%d')}, to = {df.index[-1].strftime('%Y-%m-%d')}")
            df.to_excel(writer,sheet_name=key,index=True)
        first_data_date.to_excel(writer,sheet_name='first_data_date',index=True)
        
    print(f'{args.output_file} created/updated on {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    sys.stdout = sys.__stdout__
    
def process_quarterly(qVar, QSTART, QEND, first_data_date):
    
    ## load raw data && create mapping between series and tcode
    qdraw = pd.read_csv(f'{args.folder}/{args.QFILE}').dropna(how='all')
    ## initial formatting and filtering
    qd = qdraw.iloc[args.QSTART_ROW:,:].copy()
    qd['sasdate'] = pd.to_datetime(qd['sasdate'])
    qd.set_index(['sasdate'],inplace=True)
    qd.index.name = 'timestamp'
    qd.index = qd.index.to_period('Q').to_timestamp('Q') ## use quarter-end as timestamp
    
    #print(qd.head())
    
    if first_data_date is not None:
        first_data_date.extend([{'MNEMONIC': col, 'FIRST_DATA_DATE': qd[col].first_valid_index().strftime('%Y-%m-%d'), 'freq': 'Q'} for col in qd.columns])
    
    qd = qd.filter(items=qVar) ## filter to variable of interest
    missing_qvar = list(set(qVar) - set(list(qd.columns)))
    if missing_qvar:
        warnings.warn(f'the following quarterly variabes are missing: {missing_qvar}')
        
    qd_lvls_raw = qd.copy()
    
    ## 1.2 handle missing value
    for col in qd.columns.to_list():
        if any(qd[col].isna()):
            missing_dates = list(map(lambda x: x.strftime("%Y-%m-%d"), qd.index[qd[col].isna()].to_list()))
            if len(missing_dates):
                print(f' >> {col} ({variable_dict[col]}) has NA values on {missing_dates}')
                qd[col].ffill(inplace=True)
    qd_lvls = qd
    
    ## 1.3 apply transformation
    qd_transf = {}
    for col in qd.columns:
        qd_transf[col] = trans_fn(qd[col],trans_map[col],freq=4)
    qd_transf = pd.DataFrame(qd_transf)
    qd_transf = qd_transf[(qd_transf.index >= QSTART) & (qd_transf.index <= QEND)]
    print(f'FIRST DATA DATE = {qd_transf.index[0]}, LAST DATA DATE = {qd_transf.index[-1]}')

    ## 1.4 double check missing value
    for col in qd_transf.columns.to_list():
        if any(qd_transf[col].isna()):
            missing_dates = list(map(lambda x: x.strftime("%Y-%m-%d"), qd_transf.index[qd_transf[col].isna()].to_list()))
            if len(missing_dates):
                warnings.warn(f' !! {variable_dict[col]} ({col}) has NA values; this should not happen')
    
    return qd_lvls_raw, qd_lvls, qd_transf
    
def process_monthly(mVar, MSTART, MEND, first_data_date):
    
    ## load raw data && create mapping between series and tcode
    mdraw = pd.read_csv(f'{args.folder}/{args.MFILE}').dropna(how='all')
    ## remove the first two rows, set index and apply the transformation
    md = mdraw.iloc[args.MSTART_ROW:,:].copy()
    md['sasdate'] = pd.to_datetime(md['sasdate'])
    md.set_index(['sasdate'],inplace=True)
    md.index.name = 'timestamp'
    md.index = md.index.to_period('M').to_timestamp('M') ## use month-end as timestamp
    
    #print(md.head())
    
    if first_data_date is not None:
        first_data_date.extend([{'MNEMONIC': col, 'FIRST_DATA_DATE': md[col].first_valid_index().strftime('%Y-%m-%d'), 'freq': 'M'} for col in md.columns])
        
    md = md.filter(items=mVar) ## filter to variable of interest
    missing_mvar = list(set(mVar) - set(list(md.columns)))
    if missing_mvar:
        warnings.warn(f'the following monthly variabes are missing: {missing_mvar}')

    md_lvls_raw = md.copy()
    
    ## handle missing value
    for col in md.columns.to_list():
        if any(md[col].isna()):
            missing_dates = list(map(lambda x: x.strftime("%Y-%m-%d"), md.index[md[col].isna()].to_list()))
            if len(missing_dates):
                print(f' >> {col} ({variable_dict[col]}) has NA values on {missing_dates}')
                if col in args.MFILL.keys():
                    for date, val in args.MFILL[col].items():
                        md.at[pd.to_datetime(date),col] = val
                md[col].ffill(inplace=True)
    
    md_lvls = md.copy()
    
    ## apply transformation
    md_transf = {}
    for col in md.columns:
        md_transf[col] = trans_fn(md[col],trans_map[col],freq=12)
    md_transf = pd.DataFrame(md_transf)
    md_transf = md_transf[(md_transf.index >= MSTART) & (md_transf.index <= MEND)]
    print(f'FIRST DATA DATE = {md_transf.index[0]}, LAST DATA DATE = {md_transf.index[-1]}')

    for col in md_transf.columns.to_list():
        if any(md_transf[col].isna()):
            missing_dates = list(map(lambda x: x.strftime("%Y-%m-%d"), md_transf.index[md_transf[col].isna()].to_list()))
            if len(missing_dates):
                warnings.warn(f' >> {col} ({variable_dict[col]}) has NA values on {missing_dates}; this should not happen\n')
    
    return md_lvls_raw, md_lvls, md_transf
    
def trans_fn(x,tcode,freq=4):
    """
    Performs transformation to the original series based on transformation code
    
    Arguments:
    x: pd.Series -- the raw time series to be processed
    tcode: int -- transformation code
    freq: frequency
    
    Returns:
    ret -- transformed x, of type pd.Series
    """
    ## x: pd.Series -> the raw time series to be processed
    ## tcode: int -> transformation code
    if tcode == 1:
        ret = x
    elif tcode == 2: # diff
        ret = x.diff()
    elif tcode == 3: ## twice diff
        dx = x.diff()
        ret = dx.diff()
    elif tcode == 4: ## log
        ret = np.log(x)
    elif tcode == 5: ## 100\delta log (approx for pct change)
        logx = np.log(x)
        ret = 100 * logx.diff()
    elif tcode == 6: ## twice delta log (change in growth rate)
        logx = np.log(x)
        dlogx = logx.diff()
        ret = 100 * dlogx.diff()
    elif tcode == 7: ## diff of pct change (change in growth rate)
        pct_change = x.pct_change()
        ret = 100 * pct_change.diff()
    elif tcode == 8: ## QoQ AR
        pct_change = x.pct_change()
        ret = 100 * ((1+pct_change)**4-1)
    elif tcode == 9: ## YoY
        ret = 100 * (x.pct_change(freq))
    elif tcode == 10:
        ret = x / 10.0
    elif tcode == 11:
        ret = x / 100.0
    else:
        raise ValueError(f'tcode = {tcode}, unrecognized')
    return ret
    
if __name__ == "__main__":

    print('=============================================================')
    print(f'>>> {sys.argv[0]} started execution on {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}; CWD={os.getcwd()}')
    main()
    print(f'>>> {sys.argv[0]} finished execution on {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')
