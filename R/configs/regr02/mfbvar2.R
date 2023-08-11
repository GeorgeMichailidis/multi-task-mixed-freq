
ds_identifier = 'regr02_5005'
ds_name = sprintf("../data_sim/%s.xlsx", ds_identifier)
freq_ratio = 3

lags = 4
horizon = 4

eval_metric = 'median_mape_by_step'

## these argv will be overriden in the case of CMD run
sample_size = 1000
batch_size = 100 ## number of forecasting experiments
output_filename = sprintf('output/%s/mfbvar2_%d_%d.xlsx',ds_identifier,sample_size,0)
