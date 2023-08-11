#!/bin/bash

pwd; hostname

config_filename=configs/FRED/mfbvar2.R
#module load R/4.1

date
echo "running mfbvar for fred data using config ${config_filename}"

Rscript --vanilla mfbvar_on_fred.R --config="$config_filename"
date
