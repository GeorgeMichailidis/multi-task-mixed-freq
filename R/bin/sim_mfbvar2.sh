#!/bin/bash

pwd; hostname

ds_str=ss01

#module load R/4.1

config_file_name="configs/${ds_str}/mfbvar2.R"
for sample_size in 500 1000 3000
do
    for batch_id in {1..5}
    do
        date
        CMD="Rscript --vanilla mfbvar_on_sim2.R --config=$config_file_name --sample_size=$sample_size --batch_id=$batch_id --batch_size=20"
        echo "${CMD}"
        eval ${CMD}
    done
done

date
