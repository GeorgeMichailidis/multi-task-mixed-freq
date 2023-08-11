#!/bin/bash

cd ../
pwd; hostname; date;

# to simulate a single dataset
ds_str=ss00
# to simulate multiple datasets at the same time
# ds_str='ss00,ss01,ss02,regr01,regr02'

echo "simulating datasets ${ds_str}"

CMD="python -u generate_data.py --sample_size=30000 --datasets=${ds_str}"
eval ${CMD}

date;
