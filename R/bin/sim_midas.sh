#!/bin/bash

ds_str=ss00
method=almon

if [[ ${model} != 'almon' && ${model} != 'nbeta' && ${model} != 'unres' ]]; then
    echo 'invalid method; method must be one of [almon, nbeta, unres]'
    exit 1
fi

pwd; hostname
#module load R/4.1

config_file_name="configs/${ds_str}/${method}.R"
for sample_size in 500 1000 3000
do
    for batch_id in {1..5}
    do
        date
        CMD="Rscript --vanilla midasr_on_sim.R --config=${config_file_name} --sample_size=${sample_size} --batch_id=${batch_id} --batch_size=20"
        echo "${CMD}"
        eval ${CMD}
        
    done
done

date
