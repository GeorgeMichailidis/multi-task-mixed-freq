#!/bin/bash

cd ../
pwd; hostname; date;

model=seq2seq

if [[ ${model} != 'seq2seq' && ${model} != 'seq2one' && ${model} != 'transformer' && ${model} != 'mlp' && ${model} != 'gbm' ]]; then
    echo 'invalid model; model must be one of [seq2seq, seq2one, transformer, mlp, gbm]'
    exit 1
fi

config_filename="configs/electricity/${model}.yaml"

CMD="python run_real.py --config=${config_filename} --mode=static --verbose=1"
echo "running model ${model} in static mode"
eval ${CMD}
