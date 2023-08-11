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
registerDoParallel(cores=detectCores(all.tests = FALSE, logical = TRUE)-1);
#registerDoParallel(cores=as.integer(Sys.getenv("SLURM_CPUS_ON_NODE")));

################################
## PARSE ARGV FROM CMD
################################

option_list = list(
  make_option(c("-c", "--config"), type="character", default='./configs/FREd/almon.R',help="config file", metavar="character")
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
x_step_tags = paste0(rep("step_",freq_ratio*horizon),1:(freq_ratio*horizon))
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
 
    XEND = which(x_data[,1]==prev_QE_dates[idx])
    x_train = x_data[1:XEND,-1]
    
    y_pred_by_mode = list()
    for (MODE_ID in mode_indices)
    {
        mode_specs = fn_create_specs(MODE_ID, lag_x, XEND, freq_ratio=freq_ratio,horizon=horizon)
        print(sprintf("   @prev_QE = %s, MODE_ID = %d || xlag_specs = %s", prev_QE_dates[idx], MODE_ID, mode_specs$lag_specs))
      
        ## convert data to the desired format
        datalist = xstartlist = list()
        ## create expression for evaluation
        xexpr = ""
        ## set aside new data for forecast/nowcast
        newdata = list()
        
        y_fcast = array(0,c(ncol(y_train),horizon))
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
              
              y_fcast[y_col_id,] = as.numeric(fc$mean)
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
            
            y_fcast[y_col_id,] = as.numeric(fc$mean)
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
            
            y_fcast[y_col_id,] = as.numeric(fc$mean)
          }
        } ## end of branch for restriction_type == 'unrestricted'
        
        colnames(y_fcast) = y_step_tags
        y_pred = array(0,c(ncol(y_train),3))
        y_pred[,1] = prev_QE_dates[idx]
        y_pred[,2] = mode_tags[MODE_ID+1]
        y_pred[,3] = colnames(y_train)
        colnames(y_pred) = c('prev_QE','tag','variable_name')
        y_pred = cbind(data.frame(y_pred), data.frame(y_fcast))
        
        y_pred_by_mode[[MODE_ID+1]] = y_pred
    }
    
    list(y_pred=do.call('rbind',y_pred_by_mode))
}
cat(sprintf(" >> [%s] Ended running, length of output_list = %d\n", format(Sys.time(), "%m%d%y %X"), length(output_list)))

################################
## COLLECT RESULTS
################################
## collect results for the parallel run
y_list = list()
for(idx in 1:length(output_list))
{
  y_list[[idx]] = output_list[[idx]]$y_pred
} 
summary = list(y_prediction = do.call('rbind',y_list))

############## write to result file ###############
write_xlsx(summary,path=output_filename)

