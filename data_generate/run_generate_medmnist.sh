#!/bin/bash

medmnist_path="/home/project/dfq/medmnist"
imagenet_path="/home/dataset/imagenet"
datasets=(
    'dermamnist'
    'tissuemnist'
    'pathmnist'
    'bloodmnist'
)

for dataset in "${datasets[@]}"; do
    python generate_data.py \
        --model=resnet18 \
        --dataset=${dataset} \
        --teacher_checkpoint=${medmnist_path}/${dataset}/resnet18_224_1.pth \
        --dataset_path=${imagenet_path} \
        --output_dir=../data/${dataset} \
        --file_prefix=resnet18_${dataset}_unified_curated \
        --batch_size=1024 \
        --num_augmentations=5 \
        --w_sens=0.5 \
        --w_pot=0.5 \
        --total_candidate_pool=25600 \
        --total_samples=5120 \
        --num_groups=4
done
