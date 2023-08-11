# Multi-Task Encoder-Dual-Decoder for Mixed Frequency Data Prediction 

This repository hosts the code used in the paper titled "A Multi-Task Encoder-Dual-Decoder Framework for Mixed Frequency Data Prediction". 

(Copyright 2023) Jiahe Lin and George Michailidis

# Setup

Assume miniconda or anaconda has already been installed. To set up the environment, proceed with the following commands:
```console
conda create -n mixfreq python=3.9
conda activate mixfreq
conda install pyyaml numpy pandas statsmodels scikit-learn lightgbm
conda install tensorflow pytorch-lightning pytorch
conda install matplotlib seaborn openpyxl
pip install darts
```
To verify that your GPU is up and running:
```console
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.cuda.is_available())"
```
Some useful links in case the GPU is not configured correctly:
* https://www.tensorflow.org/install/pip
* https://pytorch.org
* https://lightning.ai/docs/pytorch/stable//starter/installation.html


# Outline of the Repo

To facilitate users in traversing the repository, we highlight the major components.
* `bin/`: hosts the bash scripts where data are generated/gathered and models training/forecasting jobs are run. See experiment sections below for how to use them.
* `models/`: hosts the implementation of the models under the proposed multi-task framework (including multi-step forecasts models seq2seq, transformer and one-step forecast models seq2one), as well as other benchmark models (LightGBM, MLP) and univariate ones (ARIMA, Simple Exponential Smoother and the Naive method) under subfolder `models/benchmarks/`.
* `helpers/`: hosts the helper and utility scripts that facilitate running the experiments systematically.
* `configs/`: hosts the yaml files that stored the configuration used for running all the experiments in python.
* `R/`: hosts the R implementation of midasr and mfbvar as competing methods for running synthetic data experiments. In particular, the corresponding configs are saved under `R/configs` and please refer to `R/bin` to see how to run them. 
* `data_*/`: currently hosts the scripts used for pre-processing the underlying raw data. Note that the data themselves are not included in this repository; in the experiments, these folders should also be the location where the raw and the cleaned data are saved. Follow the instructions in the experiment sections to see how to generate or download/process the data. 
* `output_*/`: currently hosts the jupyter-notebooks where results are being collected and displayed. For the sake of completeness, in `output_sim/`, we also include the reported metrics (see Tables 2 \& 3 in the manuscript) in xlsx format. These folders are also where the experiment outputs are directed to.

# Synthetic Data Experiments

The following set of commands allows one to run synthetic data experiments considered in the paper. Toggle the specification in the bash scripts to run the corresponding data setting (i.e., specify `ds_str` and choose among ss00, ss01, ss02, regr01, regr02) and model setting (i.e., choose the corresponding scripts and specify `model` to run models under the proposed multi-task framework or competing methods). 

```console
cd bin/

## generate data; data will be saved to datafiles_sim/.; currently set to ss00
bash 00-sim-data.sh

## run the proposed multi-task framework (seq2seq, transformer, seq2one) and competing methods including MLP and LightGBM; currently set to run ss00 using multi-task seq2seq
bash 01-run-sim.sh 

## run competing methods based on NHiTs and DeepAR, leveraging the implementation in Darts; currently set to run ss00 using nhits
bash 02-run-simDarts.sh 

## run competing methods where each individual series are modeled through univariate methods (arima, simple exponential smoothing and naive); currently set to run ss00 using ses
bash 03-run-simUni.sh 
```
* A folder will be created under `output_sim/.` as the model is being trained and the forecast results being generated. The train and validation loss are saved down as a png in the folder, together with some other model information (e.g., architecture) and randomly selected columns for visualization of the fit. Raw error metrics are saved as an excel file, evaluated according to the description in the manuscript.  
* **Note that in the paper, we report the normalized error, that is, the raw error metric normalized by that of the simple exponential smoother.**
* Refer to `output_sim/collect_sim_results.ipynb` for compiling the results and generating the reported metrics in the tables presented in the paper.

# Real Data Experiments

## US Macroeconomic Dataset

The raw data are retrived from the FRED database https://research.stlouisfed.org/econ/mccracken/fred-databases/ and requires some pre-processing and cleaning. 

### Data pre-processing and cleaning
* Download data, which collects data for the majority of the variables as YYYYMM\_MRaw.csv and YYYYMM\_QRaw.csv
    ```console
    cd bin/
    ## currently vintage is set to 202207, corresponding to the one in the manuscript
    bash 10-fred-data-wget.sh 
    ```
    Two variables that are used in running mfbVAR are not present in the monthly/quarterly compilation and need to be downloaded separately; their data should be added as new columns to the monthly/quarter raw csv files, respectively.
    * AWHI (monthly): https://fred.stlouisfed.org/series/AWHI
    * FPIC1 (quarterly): https://fred.stlouisfed.org/series/FPIC1
    * FPIC1 (from BEA, used for recast): https://apps.bea.gov/iTable/?isuri=1&reqid=19&step=4&categories=flatfiles&nipa_table_list=1 (Section 5, sheet T50303-Q, Line 9)

    Note that FPIC1 needs to be recasted the same way as described in the mfbVAR paper. 

    The above step gives rise to the edited files YYYYMM\_MRaw.csv and YYYYMM\_QRaw.csv, in which additional columns have been added. 

* Perform necessary data cleaning, including fill in missing values and applying transformation:
    ```console
    bash 11-fred-data-clean.sh
    ```

    For the sake of completeness, the data after cleaning and transformation has been included in the repository as `data_FRED/202207.xlsx`. 

### Run model

To run model and gather output
```console
## toggle accordingly to switch among supported methods
bash 12-runfred.sh
```
* Similar to the case of synthetic data experiments, as the model is being trained, files (e.g., loss over epoch, model summary etc) will be saved to the designated folder under `output_FRED/` for easier diagnostics. Meanwhile, raw prediction files are saved as well. 
* Output is collected and plots are generated through `output_FRED/collect_FRED_results.ipynb`.

## Electricity Dataset

### Data pre-processing and cleaning
* Raw data is downloaded from https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather; unzip and save the folder as `data_electricity/raw`
* Refer to `data_electricity/clean_electricity.ipynb` for data cleaning

### Run model
```console
## toggle accordingly to switch among supported methods
bash 20-runelec.sh
```
* Output is collected and displayed in `output_electricity/collect_elec_results.ipynb`