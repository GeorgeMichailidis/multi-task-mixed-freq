rm(list=ls())

################################
## LOAD PACKAGES
################################

pkgs = c("mfbvar","doParallel","writexl","readxl","reshape2","optparse");
new = pkgs[!(pkgs%in%installed.packages()[,"Package"])]
if (length(new))
{ 
  for(pkg in new) 
    install.packages(pkg, dependencies = TRUE,repos = "http://cran.us.r-project.org");
}
sapply(pkgs, require, character.only = TRUE);
cores=ifelse(Sys.getenv("SLURM_CPUS_ON_NODE")!='',as.integer(Sys.getenv("SLURM_CPUS_ON_NODE")),detectCores(all.tests = FALSE, logical = TRUE)-1);
registerDoParallel(cores=cores);

################################
## PARSE ARGV FROM CMD
################################

option_list = list(
  make_option(c("-c", "--config"), type="character", default='configs/ss00/mfbvar2.R',help="config file", metavar="character"),
  make_option(c("-s", "--sample_size"), type="character", default=1000, help='training sample size', metavar="character"),
  make_option(c("-i", "--batch_id"),  type="character", default='1', help='experiment batch id', metavar="character"),
  make_option(c("-z", "--batch_size"), type="character", default='20',help='experiment batch size', metavar="character")
);
opt = parse_args(OptionParser(option_list=option_list));

################################
## GENERAL SETUPS
################################

source(opt$config)
sample_size = as.integer(opt$sample_size);
batch_id = as.integer(opt$batch_id)
batch_size = as.integer(opt$batch_size)
offset = batch_size * (batch_id - 1)

output_filename = sprintf('output/%s/mfbvar2_%d_%d.xlsx',ds_identifier,sample_size,offset)

cat('*************************************\n')
cat(sprintf('** sample_size = %d, batch_id = %d, batch_size = %d, offset = %d\n',sample_size, batch_id, batch_size, offset))
cat(sprintf('** eval metric = %s\n', eval_metric))
cat(sprintf('** output filename = %s\n', output_filename))
cat('*************************************\n')

# forecast has id = 0, nowcast vintages have their corresponding id
mode_indices = 0:freq_ratio
mode_tags = c('F',paste0(rep("N",length(mode_indices)-1),mode_indices[-1]))
x_step_tags = paste0(rep("step_",horizon*freq_ratio),1:(horizon*freq_ratio))
y_step_tags = x_step_tags[1:horizon]

################################
## LOAD DATA
################################
## read in data and basic init; note that the first col is the timestamp
x_data = data.frame(read_excel(ds_name, sheet = "x"))
y_data = data.frame(read_excel(ds_name, sheet = "y"))

################################
## LOOP THROUGH MODES
################################
TEST_ID_IN_USE = 1 + offset
cat(sprintf(" >> [%s] Started running; TEST_ID_IN_USE = %d\n", format(Sys.time(), "%m%d%y %X"), TEST_ID_IN_USE))

YSTART = TEST_ID_IN_USE
YEND = TEST_ID_IN_USE + sample_size - 1
y_train = y_data[YSTART:YEND,-1]

XSTART0 = freq_ratio*(TEST_ID_IN_USE - 1) + 1
XEND0 = XSTART0 + freq_ratio * sample_size - 1

output_list = foreach(MODE_ID = 0:freq_ratio) %dopar%
{
    ## truncate x data based on vintage (MODE_ID)
    XSTART = XSTART0
    XEND = XEND0 + MODE_ID

    print(sprintf("   @TEST_ID_IN_USE = %d, MODE_ID = %d || XSTART = %d, XEND = %d, YSTART = %d, YEND = %d\n", TEST_ID_IN_USE, MODE_ID, XSTART, XEND, YSTART, YEND))

    x_train = x_data[XSTART:XEND,-1]

    ## convert data to the desired format
    datalist = list()
    for (i in 1:ncol(x_train)){
      col = colnames(x_data)[i+1]
      datalist[[col]] = ts(x_train[,i],start=1900,frequency=12)
    }
    for (i in 1:ncol(y_train)){
      col = colnames(y_data)[i+1]
      datalist[[col]] = ts(y_train[,i],start=1900,frequency=4)
    }
  
    ## use default prior, estimate, predict
    prior_obj = set_prior(Y = datalist, n_lags = lags, n_reps = 20, n_fcst=horizon*freq_ratio-MODE_ID)
    mod_minn = estimate_mfbvar(prior_obj, prior = "minn")
    pred = data.frame(predict(mod_minn))
    
    ## format the predictions
    x_fcast = array(0,c(horizon*freq_ratio,ncol(x_train)))
    for (i in 1:ncol(x_train))
    {
      temp = pred[pred$variable==sprintf('x%d',i-1),]
      x_fcast[,i] = temp[,'median']
    }
    colnames(x_fcast) = colnames(x_train)
    
    y_fcast = array(0,c(horizon,ncol(y_train)))
    for (i in 1:ncol(y_train)){
      temp = pred[pred$variable==sprintf('y%d',i-1),]
      ## note that in the case of forecast/N1, temp would include the last quarter actuals
      ## in the case of N2/N3, temp would not include the last available actuals
      y_fcast[,i] = temp[(nrow(temp)-horizon+1):(nrow(temp)),'median']
    }
    colnames(y_fcast) = colnames(y_train)
    
    ## remove variable so that mem can be released??
    rm(list=c('x_train','datalist','prior_obj','mode_minn','pred'))
    
    ## record deviation from the truth
    y_true = y_data[(YEND+1):(YEND+horizon),-1]
    x_true = x_data[(XEND0+1):(XEND0+horizon*freq_ratio),-1]
    
    if(eval_metric == 'median_mape_by_step')
    {
        x_err = as.numeric(apply(abs(x_fcast - x_true)/abs(x_true),1,function(x){median(x)}))
        y_err = as.numeric(apply(abs(y_fcast - y_true)/abs(y_true),1,function(x){median(x)}))
    }
    if (eval_metric == 'rmse_by_step')
    {
        x_err = as.numeric(apply(x_fcast - x_true,1,function(x){sqrt(mean(x^2))}))
        y_err = as.numeric(apply(y_fcast - y_true,1,function(x){sqrt(mean(x^2))}))
    }
    list(x_err=x_err,y_err=y_err)
}
cat(sprintf(" >> [%s] Ended running, length of output_list = %d\n", format(Sys.time(), "%m%d%y %X"), length(output_list)))

################################
## COLLECT RESULTS
################################
## initialization of the forecast and the error matrix
x_err = array(NA,c(length(mode_indices),horizon*freq_ratio))
y_err = array(NA,c(length(mode_indices),horizon))

for(MODE_ID in 0:freq_ratio)
{
  x_err[MODE_ID+1,] = output_list[[MODE_ID+1]]$x_err
  y_err[MODE_ID+1,] = output_list[[MODE_ID+1]]$y_err
} 
x_err = data.frame(mode_tag=mode_tags,experiment_id=TEST_ID_IN_USE,data.frame(x_err))
colnames(x_err) = c('mode','experiment_id',x_step_tags)

y_err = data.frame(mode_tag=mode_tags,experiment_id=TEST_ID_IN_USE,data.frame(y_err))
colnames(y_err) = c('mode','experiment_id',y_step_tags)

err_summary = list(x_err=x_err, y_err=y_err)

############## write to result file ###############
write_xlsx(err_summary,path=output_filename)

