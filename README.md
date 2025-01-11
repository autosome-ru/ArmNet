# ✋ArmNet

[`CURRENTLY UNDER CONSTRUCTION`]

Here we present a hybrid transformer and convolutional network for predicting RNA reactivities based on data provided in [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) competition.
<br>

Our approach took the [first place](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460121) in the Ribonanza competition. In this repository we present our post-competition model based on methods used by other contestants and our insights. 
<br>
[[`Ribonanza Paper Preprint`](https://www.biorxiv.org/content/10.1101/2024.02.24.581671v1)]

## 👩‍💻Model weights
The ArmNet weights are available at [Kaggle](https://www.kaggle.com/datasets/vyaltsevvaleriy/armnet-weights/data?select=train_test_pseudolabel_weights).

## 🏃‍♀️Training the model

First, download data from [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data) competition and then specify data paths in config file. If you want to train model which takes BPPMs (base pair probability matrices) as input you should do the fllowing:

* for train matrices merge all matrices in file named "joined.mmap" using `numpy.memmap` and save array of corresponding sequence IDs in file named "index.ind" using `numpy.save`
* for test matrices save individual matrices using `numpy.save` in separate directory; these files should be named as "SEQUENCE_ID.npy", SEQUENCE_ID is corresponding sequence's ID

For each sequence in train and test datasets we predicted BPPM using [EternaFold](https://github.com/eternagame/EternaFold) package.

All data paths along with main model parameters are specified in **config.json** file. Architectural parameters can be changed only through config.json, whereas training parameters can be changed as command line arguments. By default, in each run all scripts save config-file with all parameters used.

The examples of running training script:
```
python train.py -c CONFIG.json

# Output directory should be specified as command line argument; be default all output files are saved in directory "results"
python train.py -c CONFIG.json --out_dir_path ./chosen_dir

# Training parameters can be conviniently changed in command line
# Values specified in command line will be used instead of corresponding values from config-file
# By default, all changes are saved in separate config file in ouput directory 
python train.py -c CONFIG.json --out_dir_path ./chosen_dir --device 0 --max_lr 0.002
```

## 🕵️‍♂️Testing on competition data
Analogously to training script, `test.py` uses config file, but it requires model weights specified in `--pretrained_model_weights` argument. This script generates contest submission.

The examples of running testing script:
```
python test.py -c CONFIG.json --pretrained_model_weights ./results/sgd_run/model.pth
```

To convert submisiion file to dataframe of format equivalent to training dataset you can use `to_train_format` function from `test_utils.py`

## 🧙‍♂️Predicting on custom data
If you want to predict reactivities with ArmNet for your data, you have to options: using an ensemble model that requires calculating BPPMs (1) and using single model trained on ensemble predictions and which doesn't require calculating BPPMs (2). Both options require that data be converted to a `.tsv` file with two columns: `id` - unique id for each sequence, `sequence` - nucleotide sequence of RNA (with `T` instead of `U`) (as in `example_data.tsv`).

### 1. Predicting with ensemble with BPPMs
Firstly, you need to install [`arnie`](https://github.com/DasLab/arnie) package. Then in `get_bppms.py` specify paths to arnie config and arnie directory (`arnie/src`) and run the script as shown in the example below
```
python3 get_bppms.py -o path/to/bppm_directory -d path/to/data.tsv
```
Then run `predict_ensemble_bppm.py` with BPPM data available. Here's example of how to run this script

```
python3 predict_ensemble_bppm.py -o path/to/output_directory -d path/to/data.tsv -c CONFIG.json --pretrained_models_dir path/to/ensemble_weights --data_bppm_path path/to/bppm_directory
```
Predicted reactivities will be available in `ensembled` directory within output directory in numpy-memmap format (shape of array and encoding are specified in the names of files). The order of items in array corresponds to order of sequences in initial dataframe.

### 2. Predicting with single model without BPPMs
Here we can skip the step with calculating BPPMs and just run `predict_nobppm.py` as follows

```
python3 predict_nobppm.py -o path/to/output_directory -d path/to/data.tsv -c CONFIG.json --pretrained_models_dir path/to/train_test_pseudolabel_weights
```
Here you can use `train_test_pseudolabel_weights` and `train_pseudolabel_weights` weights from Kaggle.
