#!/bin/bash

cd ../
pwd; hostname; date;

vintage=202207

mkdir -p logs/FRED

CMD="python data_FRED/clean_fred.py --config=configs/FRED/data_preproc/${vintage}.yaml"
eval ${CMD}
