# Relaxed Label Shift Study Code

This readme gives a brief overview of the code. We will release a detailed README with the final version of the paper. 

## Overview

Overall pipeline: 

The following are the crucial parts of the code:

1. `label_shift_utils.py`: This files contains utils functions to simulate label shift in the target data. 
2. `./datasets/get_dataset.py`: This file contains the code to get the source and target datasets.
3. `./algorithms/`: This folder contains the code for different algorithms. We implement the follwing domain algorithms:  
    - ERM variants: ERM, ERM-aug, with different pretraining techniques like ['rand', 'imagenet', 'clip']
    - Domain alignment methods: DANN, CDAN,
    - Self-training methods: Noisy student, Pseudolabeling, FixMatch, SENTRY, COAL 
    - Test time adaptation methods: BN_adapt, TENT,  CORAL 


The entry point of the code is `run_expt.py`. `config` folder contains default parameters and hyperparameters needed for base experiments for the project. We need to pass the dataset name and the algorithm name with flag `--dataset` and `--algorithm` to the `run_expt.py` file. To simulate label shift, we need to pass the flag `--simulate_label_shift` and the dirchilet sampling parameter with `--dirchilet_alpha`. And the flag `--root_dir` is used to specified the data directory for source and target datasets.

Caveat: For Test Time Adaptation (TTA) methods, we need to provide the folder with ERM-aug trained models with the parameter `--source_model_path`.

### Results Logging 

The code evaluates the models trained and logs the results in the `./logs/` folder in form of a csv file. 


## Simple example for running the code
The following command can be used to run the code on `cifar10` dataset with `ERM-aug` algorithm:

```python
python run_expt.py --dataset=cifar10 --algorithm=ERM-aug --simulate_label_shift --dirchilet_alpha=0.1
```

## Requirements 

The code is written in Python and uses [PyTorch](https://pytorch.org/). To install requirements, setup a conda enviornment using the following command:

```setup
conda env create --file requirements.yml
```

## Dataset Setup 
To setup different datasets, run the scrips in `dataset_scripts` folder. Except for Imagenet dataset which can be downloaded from the [official website](https://www.image-net.org/download.php), the scripts set up all the datasets (including all the source and target pairs) used in our study.

