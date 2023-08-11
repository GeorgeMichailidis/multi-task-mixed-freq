#!/bin/bash

cd ../
pwd; hostname; date;

## download raw data from fred
vintage=202207


odoc_M="./data_FRED/${vintage}_Mraw.csv"
odoc_Q="./data_FRED/${vintage}_Qraw.csv"
wget --output-document $odoc_M "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/${vintage}.csv"
wget --output-document $odoc_Q "https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/${vintage}.csv"
