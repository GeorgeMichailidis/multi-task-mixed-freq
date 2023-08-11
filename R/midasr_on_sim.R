rm(list=ls())

################################
## LOAD PACKAGES
################################

pkgs = c("midasr","doParallel","writexl","readxl","reshape2","optparse");
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
  make_option(c("-c", "--config"), type="character", default='configs/ss00/almon.R',help="config file", metavar="character"),
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

output_filename = sprintf('output/%s/%s_%d_%d.xlsx',ds_identifier,restriction_type,sample_size,offset)

cat('*************************************\n')
cat(sprintf('** sample_size = %d, batch_id = %d, batch_size = %d, offset = %d\n',sample_size, batch_id, batch_size, offset))
cat(sprintf('** eval metric = %s\n', eval_metric))
cat(sprintf('** output filename = %s\n', output_filename))
cat('*************************************\n')

# forecast has id = 0, nowcast vintages have their corresponding id
mode_indices = 0:freq_ratio
mode_tags = c('F',paste0(rep("N",length(mode_indices)-1),mode_indices[-1]))
x_step_tags = paste0(rep("step",freq_ratio*horizon),1:(freq_ratio*horizon))
y_step_tags = x_step_tags[1:horizon]

# helper function for creating lag specification & new data rows for each run
fn_create_specs = function(MODE_ID, lag_x, XEND, freq_ratio=3, horizon=1)
{
  lag_specs = sprintf('%s:%s',freq_ratio-MODE_ID, freq_ratio+lag_x-MODE_ID-1)
  newdata_idx = (XEND-freq_ratio+1+MODE_ID):(XEND+MODE_ID+freq_ratio*(horizon-1))
  return(list(lag_specs = lag_specs, newdata_idx = newdata_idx))
}

################################
## LOAD DATA
################################
## read in data and basic init; note that the first col is the timestamp
x_data = data.frame(read_excel(ds_name, sheet = "x"))
y_data = data.frame(read_excel(ds_name, sheet = "y"))

################################
## LOOP THROUGH EXPERIMNETS
################################
cat(sprintf(" >> [%s] Started running\n", format(Sys.time(), "%m%d%y %X")))
output_list = foreach(TEST_ID = 1:batch_size) %dopar%
{
    TEST_ID_IN_USE = TEST_ID + offset
    
    XSTART = freq_ratio*(TEST_ID_IN_USE-1) + 1
    XEND = XSTART + freq_ratio * sample_size - 1
  
    YSTART = TEST_ID_IN_USE
    YEND = YSTART + sample_size - 1
  
    x_train = x_data[XSTART:XEND,-1]
    y_train = y_data[YSTART:YEND,-1]
    
    ## truth to be evaluate against
    y_true = y_data[(YEND+1):(YEND+horizon),-1]
    ## error recording
    y_err = array(NA,c(horizon,length(mode_indices)))
    
    for (MODE_ID in mode_indices)
    {
        #cat(sprintf("   @TEST_ID = %d, MODE_ID = %d || xstart = %d, xend = %d, ystart = %d, yend = %d\n", TEST_ID, MODE_ID, XSTART, XEND, YSTART, YEND))
      
        y_fcast = array(0,c(horizon,ncol(y_train)))
        
        mode_specs = fn_create_specs(MODE_ID, lag_x, XEND, freq_ratio=freq_ratio,horizon=horizon)
        ## convert data to the desired format
        datalist = xstartlist = list()
        ## create expression for evaluation
        xexpr = ""
        ## set aside new data for forecast/nowcast
        newdata = list()
        
        if (restriction_type=='almon')
        {
          for (ix in 1:ncol(x_train))
          {
              col = colnames(x_data)[ix+1]
              datalist[[col]] = x_train[,ix]
              xstartlist[[col]] = c(1,-0.5)
              newdata[[col]] = as.numeric(x_data[mode_specs$newdata_idx,ix+1])
        
              xexpr_to_add = sprintf("mls(%s,%s,%d,nealmon)",col,mode_specs$lag_specs,freq_ratio)
              xexpr = sprintf("%s+%s", xexpr, xexpr_to_add)
          }    
          for (y_col_id in 1:(ncol(y_data)-1))
          {
              curr_datalist = datalist
              curr_expression = xexpr
            
              ycol = colnames(y_data)[y_col_id+1]
              curr_datalist[[ycol]] = y_train[,y_col_id]
              expr_for_eval = sprintf("%s~mls(%s,1:%d,1,'*')%s",ycol,ycol,lag_y,curr_expression)
  
              mr = midas_r(eval(str2expression(expr_for_eval)),data=curr_datalist,start=xstartlist,Ofunction='optim',method='Nelder-Mead')
              fc = forecast(mr,newdata=newdata,method = "dynamic")
              
              y_fcast[,y_col_id] = as.numeric(fc$mean)
          }
        } ## end of branch for restriction_type == 'almon'
        
        if (restriction_type=='nbeta')
        {
          for (ix in 1:ncol(x_train))
          {
            col = colnames(x_data)[ix+1]
            datalist[[col]] = x_train[,ix]
            xstartlist[[col]] = c(0.5,-0.5,0.5)
            newdata[[col]] = as.numeric(x_data[mode_specs$newdata_idx,ix+1])
            
            xexpr_to_add = sprintf("mls(%s,%s,%d,nbeta)",col,mode_specs$lag_specs,freq_ratio)
            xexpr = sprintf("%s+%s", xexpr, xexpr_to_add)
          }    
          
          for (y_col_id in 1:(ncol(y_data)-1))
          {
            curr_datalist = datalist
            curr_expression = xexpr
            
            ycol = colnames(y_data)[y_col_id+1]
            curr_datalist[[ycol]] = y_train[,y_col_id]
            expr_for_eval = sprintf("%s~mls(%s,1:%d,1,'*')%s",ycol,ycol,lag_y,curr_expression)
            
            mr = midas_r(eval(str2expression(expr_for_eval)),data=curr_datalist,start=xstartlist,Ofunction='optim',method='Nelder-Mead')
            fc = forecast(mr,newdata=newdata,method='dynamic')
            
            y_fcast[,y_col_id] = as.numeric(fc$mean)
          }
        } ## end of branch for restriction_type == 'nbeta'
        
        if (restriction_type == 'unres')
        {
          for (ix in 1:ncol(x_train))
          {
            col = colnames(x_data)[ix+1]
            datalist[[col]] = x_train[,ix]
            newdata[[col]] = as.numeric(x_data[mode_specs$newdata_idx,ix+1])
            
            xexpr_to_add = sprintf("mls(%s,%s,%d)",col,mode_specs$lag_specs,freq_ratio)
            xexpr = sprintf("%s+%s", xexpr, xexpr_to_add)
          }    
          
          for (y_col_id in 1:(ncol(y_data)-1))
          {
            curr_datalist = datalist
            curr_expression = xexpr
            
            ycol = colnames(y_data)[y_col_id+1]
            curr_datalist[[ycol]] = y_train[,y_col_id]
            expr_for_eval = sprintf("%s~mls(%s,1:%d,1,'*')%s",ycol,ycol,lag_y,curr_expression)
            
            mr = midas_r(eval(str2expression(expr_for_eval)),data=curr_datalist,start=NULL,Ofunction='optim',method='Nelder-Mead')
            fc = forecast(mr,newdata=newdata,method='dynamic')
            
            y_fcast[,y_col_id] = as.numeric(fc$mean)
          }
        } ## end of branch for restriction_type == 'unrestricted'
        
        if(eval_metric == 'median_mape_by_step')
        {
            y_err[,MODE_ID+1] = as.numeric(apply(abs(y_fcast - y_true)/abs(y_true),1,function(x){median(x)}))
        }
        if(eval_metric == 'rmse_by_step')
        {
            y_err[,MODE_ID+1] = as.numeric(apply(y_fcast - y_true,1,function(x){sqrt(mean(x^2))}))
        }
    }
    
    y_err = data.frame(y_err)
    colnames(y_err) = mode_tags
    y_err$step_tag = y_step_tags
    
    list(y_err = y_err)
}
cat(sprintf(" >> [%s] Ended running, length of output_list = %d\n", format(Sys.time(), "%m%d%y %X"), length(output_list)))

################################
## COLLECT RESULTS
################################
## collect results for the parallel run
y_elist = list()
for(TEST_ID in 1:length(output_list))
{
  y_err = output_list[[TEST_ID]]$y_err
  y_err$experiment_id = TEST_ID
  y_elist[[TEST_ID]] = y_err
} 
y_err = do.call('rbind',y_elist)

## summarize err by tag
err_summary = list()
for (mode_tag in mode_tags){
  y_temp = subset(y_err,select=c(mode_tag,'experiment_id','step_tag'))
  err_summary[[sprintf('y_%s',mode_tag)]] = dcast(y_temp, experiment_id ~ step_tag, value.var = mode_tag)[,c('experiment_id',y_step_tags)]
}

## final calc of mean, median, std
y_summary = list()
for(tag in y_step_tags)
{
  temp = y_err[y_err$step_tag==tag,mode_tags]
  
  mtx = data.frame(array(0,c(3,length(mode_tags))))
  mtx[1,] = apply(temp,2,mean)
  mtx[2,] = apply(temp,2,median)
  mtx[3,] = sqrt(apply(temp,2,var))
  mtx$metric = c('mean','median','std')
  mtx = cbind(rep(tag,nrow(mtx)),mtx)
  colnames(mtx) = c('step',mode_tags,'metric')
  
  y_summary[[tag]] = mtx
}
y_summary = do.call('rbind',y_summary)
rownames(y_summary) = NULL
err_summary$summary_y_err = y_summary[order(y_summary$metric),]

############## write to result file ###############
write_xlsx(err_summary,path=output_filename)

