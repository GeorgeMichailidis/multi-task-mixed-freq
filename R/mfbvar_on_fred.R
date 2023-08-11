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
registerDoParallel(cores=detectCores(all.tests = FALSE, logical = TRUE)-1);
#registerDoParallel(cores=as.integer(Sys.getenv("SLURM_CPUS_ON_NODE")));

################################
## PARSE ARGV FROM CMD
################################

option_list = list(
  make_option(c("-c", "--config"), type="character", default='configs/FRED/mfbvar2.R',help="config file", metavar="character")
);
opt = parse_args(OptionParser(option_list=option_list));

################################
## GENERAL SETUPS
################################

source(opt$config)

cat('*************************************\n')
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

## read in data and basic init; note that the first col is the date/timestamp
x_data = data.frame(read_excel(ds_name, sheet = "x_small"))
y_data = data.frame(read_excel(ds_name, sheet = "y_small"))

## process date
x_data[,1] = sapply(x_data[,1],function(x){gsub(" UTC", "", x)})
y_data[,1] = sapply(y_data[,1],function(x){gsub(" UTC", "", x)})

################################
## LOOP THROUGH EXPERIMNETS
################################
cat(sprintf(" >> [%s] Started running\n", format(Sys.time(), "%m%d%y %X")))
output_list = foreach(idx = 1:length(prev_QE_dates)) %dopar%
{
  YEND = which(y_data[,1]==prev_QE_dates[idx])
  y_train = y_data[1:YEND,-1]
  
  x_pred_by_mode = y_pred_by_mode = list()
  XSTART = 1
  for (MODE_ID in mode_indices)
  {
    XEND0 = which(x_data[,1]==prev_QE_dates[idx])
    XEND = XEND0 + MODE_ID
    
    cat(sprintf("   @prev_QE = %s, MODE_ID = %d || XSTART = %s, XEND = %s, YSTART = %s, YEND = %s\n",
                prev_QE_dates[idx], MODE_ID, x_data[XSTART,1], x_data[XEND,1], y_data[1,1], y_data[YEND,1]))
    
    x_train = x_data[XSTART:XEND,-1]
    
    ## convert data to the desired format
    datalist = list()
    for (i in 1:ncol(x_train)){
      col = colnames(x_data)[i+1]
      datalist[[col]] = ts(x_train[,i],start=1960,frequency=12)
    }
    for (i in 1:ncol(y_train)){
      col = colnames(y_data)[i+1]
      datalist[[col]] = ts(y_train[,i],start=1960,frequency=4)
    }
  
    ## use default prior, estimate, predict
    prior_obj = set_prior(Y = datalist, n_lags = lags, n_reps = 20, n_fcst=horizon*freq_ratio - MODE_ID)
    mod_minn = estimate_mfbvar(prior_obj, prior = "minn")
    pred = data.frame(predict(mod_minn))
    
    ## format the predictions
    x_fcast = array(0,c(ncol(x_train),horizon*freq_ratio))
    for (i in 1:ncol(x_train))
    {
      temp = pred[pred$variable==colnames(x_train)[i],]
      x_fcast[i,] = temp[,'median']
    }
    colnames(x_fcast) = x_step_tags
    
    y_fcast = array(0,c(ncol(y_train),horizon))
    for (i in 1:ncol(y_train)){
      temp = pred[pred$variable==colnames(y_train)[i],]
      y_fcast[i,] = temp[(nrow(temp)-horizon+1):nrow(temp),'median']
    }
    colnames(y_fcast) = y_step_tags
    
    ## final touch up of the prediction matrix
    x_pred = array(0,c(ncol(x_train),3))
    x_pred[,1] = prev_QE_dates[idx]
    x_pred[,2] = mode_tags[MODE_ID+1]
    x_pred[,3] = colnames(x_train)
    colnames(x_pred) = c('prev_QE','tag','variable_name')
    x_pred = cbind(data.frame(x_pred), data.frame(x_fcast))
    
    y_pred = array(0,c(ncol(y_train),3))
    y_pred[,1] = prev_QE_dates[idx]
    y_pred[,2] = mode_tags[MODE_ID+1]
    y_pred[,3] = colnames(y_train)
    colnames(y_pred) = c('prev_QE','tag','variable_name')
    y_pred = cbind(data.frame(y_pred), data.frame(y_fcast))

    ## remove variable so that mem can be released??
    rm(list=c('x_train','datalist','prior_obj','mod_minn','pred'))
    
    x_pred_by_mode[[MODE_ID+1]] = x_pred
    y_pred_by_mode[[MODE_ID+1]] = y_pred
  }
  list(x_pred=do.call('rbind',x_pred_by_mode),
       y_pred=do.call('rbind',y_pred_by_mode)
      )
}
cat(sprintf(" >> [%s] Ended running, length of output_list = %d\n", format(Sys.time(), "%m%d%y %X"), length(output_list)))

################################
## COLLECT RESULTS
################################

## collect results for the parallel run
x_list = y_list = list()
for(idx in 1:length(output_list))
{
  x_list[[idx]] = output_list[[idx]]$x_pred
  y_list[[idx]] = output_list[[idx]]$y_pred
} 
summary = list(x_prediction = do.call('rbind',x_list), y_prediction = do.call('rbind',y_list))

############## write to result file ###############
write_xlsx(summary,path=output_filename)
