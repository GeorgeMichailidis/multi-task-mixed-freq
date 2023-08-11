
ds_identifier = 'ss02_3003'
ds_name = sprintf("../data_sim/%s.xlsx", ds_identifier)
freq_ratio = 3

restriction_type = 'nbeta'

lag_x = 12
lag_y = 2

horizon = 4

eval_metric = 'median_mape_by_step'

## these argv will be overriden in the case of CMD run
sample_size = 1000
batch_size = 100 ## number of forecasting experiments
output_filename = sprintf('output/%s/%s_%d_%d.xlsx',ds_identifier, restriction_type, sample_size, 0)
