#!/bin/bash

pwd; hostname

config_filename=configs/FRED/almon.R
#module load R/4.1

date

echo "running midasr for fred data using config ${config_filename}"

Rscript --vanilla midasr_on_fred.R --config="$config_filename"

date
