#!/bin/bash

datasets=(
    'pathmnist'
    'bloodmnist'
)

for dataset in "${datasets[@]}"; do
    python generate_data.py \
        --model=resnet18 \
        --dataset=${dataset} \
        --teacher_checkpoint=/path/to/${dataset}/resnet18_teacher.pth \
        --dataset_path=/path/to/imagenet \
        --output_dir=../data/${dataset} \
        --file_prefix=resnet18_${dataset}_unified_curated \
        --subset_size=500000 \
        --batch_size=128 \
        --num_augmentations=5 \
        --w_sens=0.5 \
        --w_pot=0.5 \
        --samples_per_class=50 \
        --num_groups=4
done
