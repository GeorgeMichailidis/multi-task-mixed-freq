
ds_name = "../data_FRED/202306.xlsx"
freq_ratio = 3

restriction_type = 'almon'

lag_x = 12
lag_y = 2

horizon = 4

prev_QE_dates = c('2017-12-31',
                  '2018-03-31','2018-06-30','2018-09-30','2018-12-31',
                  '2019-03-31','2019-06-30','2019-09-30','2019-12-31',
                  '2020-03-31','2020-06-30','2020-09-30','2020-12-31',
                  '2021-03-31','2021-06-30','2021-09-30','2021-12-31',
                  '2022-03-31','2022-06-30','2022-09-30','2022-12-31')

output_filename = sprintf('output/FRED/%s.xlsx',restriction_type)

