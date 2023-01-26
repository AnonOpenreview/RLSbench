#!/bin/bash
NUM_RUNS=15
GPU_IDS=( 1 2 3 4 5 6 7 0 ) 
NUM_GPUS=${#GPU_IDS[@]}
counter=0

DATASETS=( 'fmow' 'domainnet' 'visda' 'officehome' 'camelyon' )
SEEDS=( 42 )
ALPHA=('0.0' '0.5' '1.0' '5.0' '10.0' '100.0')
ALGORITHMS=( "ERM" "ERM-aug" )

for dataset in "${DATASETS[@]}"; do
for algorithm in "${ALGORITHMS[@]}"; do
for seed in "${SEEDS[@]}"; do

	 # Get GPU id.
	 gpu_idx=$((counter % $NUM_GPUS))
	 gpu_id=${GPU_IDS[$gpu_idx]}
	 
     cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python run_expt.py '--remote' 'False' '--dataset' ${dataset} \
	 '--root_dir' '../data' '--seed' ${seed} '--transform' 'image_none' '--algorithm'  ${algorithm}"
	 
     echo $cmd
	 eval ${cmd} &

	 counter=$((counter+1))
	 if ! ((counter % NUM_RUNS)); then
		  wait
	 fi
done
done
done

ALGORITHMS=( "ERM" )

for dataset in "${DATASETS[@]}"; do
for algorithm in "${ALGORITHMS[@]}"; do
for seed in "${SEEDS[@]}"; do

	 # Get GPU id.
	 gpu_idx=$((counter % $NUM_GPUS))
	 gpu_id=${GPU_IDS[$gpu_idx]}
	 
     cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python run_expt.py '--remote' 'False' '--dataset' ${dataset} \
	 '--root_dir' '../data' '--seed' ${seed} '--algorithm'  ${algorithm} --model 'clipvitb16' \
	 --n_epochs 10 --lr 0.00001 --weight_decay 0.1"
	 
     echo $cmd
	 eval ${cmd} &

	 counter=$((counter+1))
	 if ! ((counter % NUM_RUNS)); then
		  wait
	 fi
done
done
done


