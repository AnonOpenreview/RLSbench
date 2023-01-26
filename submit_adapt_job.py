import json
import os
import sys
import time
from collections import Counter
from datetime import date
from subprocess import Popen


NUM_RUNS=16
# GPU_IDS=list(range(8))
# NUM_GPUS=len(GPU_IDS)
counter=0

DATASETS = [
    # 'camelyon',
    # # # 'iwildcam',
    # 'fmow',
    # # # 'cifar10',
    # # # 'cifar100',
    # 'domainnet',
    'entity13', 
    'entity30',
    # # 'living17',
    # # 'nonliving26',
    # # 'office31',
    # 'officehome',
    # 'visda'
]
TARGET_SETS = {
    'cifar10': ['0','1', '10', '23','57', '71', '95'],
    'cifar100': ['0', '4', '12', '43', '59', '82'], 
    'fmow': ['0','1','2'], 
    'iwildcams': ['0','1','2'], 
    'camelyon': ['0','1','2'], 
    'domainnet': ['0','1','2','3'], 
    'entity13': ['0','1','2', '3'],
    'entity30': ['0','1','2', '3'],
    'living17': ['0','1','2', '3'],
    'nonliving26': ['0','1','2', '3'],
    'officehome': ['0','1','2', '3'], 
    'office31': ['0','1','2'], 
    'visda': ['0','1','2'], 
}

SEEDS = ['42']
ALPHA = ['0.5', '1.0',  '3.0', '10.0', '100.0']
ALGORITHMS= ["CDANN", "FixMatch"]
# ALGORITHMS= ["ERM-aug"]

procs = []

for dataset in DATASETS:
    for seed in SEEDS: 
        for alpha in ALPHA: 
            for algorithm in ALGORITHMS:             
                for target_set in TARGET_SETS[dataset]:

                    cmd=f"python submit_job.py --command 'python run_expt.py --remote True \
                    --dataset {dataset} --root_dir ./data --seed {seed} \
                    --transform image_none --algorithm  {algorithm}\
                    --dirichlet_alpha {alpha} --target_split {target_set} \
                    --use_target True  --simulate_label_shift True' \
                    --name={algorithm}_{dataset}_{seed}_{target_set}_{alpha} --timeout 90000"
                    
                    print(cmd)                    
                    procs.append(Popen(cmd, shell=True))
                    
                    time.sleep(10)

                    counter += 1

                    if counter % NUM_RUNS == 0:
                        for p in procs:
                            p.wait()
                        procs = []
                        time.sleep(10)

                        print("\n \n \n \n --------------------------- \n \n \n \n")
                        print(f"{date.today()} - {counter} runs submitted")
                        sys.stdout.flush()
                        print("\n \n \n \n --------------------------- \n \n \n \n")


for p in procs:
    p.wait()
procs = []