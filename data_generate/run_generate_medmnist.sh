#!/bin/bash
datasets=(
    # 'pathmnist'
    # 'octmnist'
    # 'pneumoniamnist'
    # 'breastmnist'
    # 'dermamnist'
    # 'bloodmnist'
	# 'retinamnist'
	# 'tissuemnist'
    # 'dermamnist'
	# 'tissuemnist'
    'pathmnist'
    'bloodmnist'
)

for dataset in ${datasets[@]}
do
	for g in 1 2 3 4
	do
	python generate_data.py 		\
			--model=resnet18 	 \
			--dataset=${dataset} \
			--image_size=224 \
			--batch_size=256 		\
			--test_batch_size=512 \
			--group=$g \
			--beta=0.1 \
			--gamma=0.5 \
			--save_path_head=../data/${dataset} \
			# --init_data_path=/home/dataset/imagenet/train

	done
done
