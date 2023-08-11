#!/bin/bash

cd ../
pwd; hostname; date;

ds_str=ss00
model=nhits
train_size=3000

if [[ ${model} != 'deepar' && ${model} != 'nhits' ]]; then
    echo 'invalid model; model must be one of [deepar, nhits]'
    exit 1
fi

config=configs/${ds_str}/${model}.yaml
echo "performing model training and forecasting for ${ds_str} based on ${config}"

CMD="python -u run_simDarts.py --config=${config} --train_size=${train_size}"
eval ${CMD}

date;
