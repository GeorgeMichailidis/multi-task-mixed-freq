#!/bin/bash

cd ../
pwd; hostname; date;

ds_str=ss00
model=ses
train_size=3000

if [[ ${model} != 'arima' && ${model} != 'naive' && ${model} != 'ses' ]]; then
    echo 'invalid model; model must be one of [arima, naive, ses]'
    exit 1
fi

config=configs/${ds_str}/${model}.yaml
echo "performing model training and forecastin for ${ds_str} based on ${config}"

CMD="python -u run_simUni.py --config=${config} --train_size=${train_size}"
eval ${CMD}

date;
