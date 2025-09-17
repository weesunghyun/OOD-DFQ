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
        --subset_size=500000 \
        --batch_size=1024 \
        --num_augmentations=5 \
        --w_sens=0.5 \
        --w_pot=0.5 \
        --samples_per_class=50 \
        --num_groups=4
done
