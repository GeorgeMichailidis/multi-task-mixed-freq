#!/bin/bash

cd ../
pwd; hostname; date;

ds_str=ss00
model=seq2seq
train_size=3000

if [[ ${model} != 'seq2seq' && ${model} != 'seq2one' && ${model} != 'transformer' && ${model} != 'mlp' && ${model} != 'gbm' ]]; then
    echo 'invalid model; model must be one of [seq2seq, seq2one, transformer, mlp, gbm]'
    exit 1
fi

config=configs/${ds_str}/${model}.yaml
echo "performing model training and forecasting for ${ds_str} based on ${config}"

CMD="python -u run_sim.py --config=${config} --train_size=${train_size} --verbose=10"
eval ${CMD}

date;
