#!/bin/bash
NUM_RUNS=4
GPU_IDS=( 0 1 ) 
NUM_GPUS=${#GPU_IDS[@]}
counter=0

DATASETS=( 'officehome' )
SEEDS=( 42 )
ALPHA=( '3.0' '10.0' '1.0' '100.0' '0.5' )
ALGORITHMS=( "IS-DANN" )
# ALGORITHMS=( "IS-NoisyStudent" )
TARGET_ARR=( '0' '1' '2' '3' )

for dataset in "${DATASETS[@]}"; do
for algorithm in "${ALGORITHMS[@]}"; do
for target_set in "${TARGET_ARR[@]}"; do
for alpha in "${ALPHA[@]}"; do
for seed in "${SEEDS[@]}"; do


	# Get GPU id.
	gpu_idx=$((counter % $NUM_GPUS))
	gpu_id=${GPU_IDS[$gpu_idx]}
	
	cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python run_expt.py '--remote' 'False' \
	'--dataset' ${dataset}  '--root_dir' '../data' '--seed' ${seed} \
	'--transform' 'image_none' --simulate_label_shift 'True' '--dirichlet_alpha' \
	${alpha} --target_split ${target_set}  '--algorithm'  ${algorithm}"

	echo $cmd

	eval ${cmd} &

	counter=$((counter+1))
	if ! ((counter % NUM_RUNS)); then
		wait
	fi
done
done
done
done 
done
