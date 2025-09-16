
import argparse
import torch
import numpy as np
import torch.nn as nn
import os
from pytorchcv.model_provider import get_model as ptcv_get_model
from distill_data import *
from collections import OrderedDict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ResNet18, ResNet50

# Classification task datasets from MedMNIST2D
CLASSIFICATION_DATASETS = {
    'pathmnist': 9,      # Colon Pathology - Multi-Class (9)
    'dermamnist': 7,     # Dermatoscope - Multi-Class (7)
    'octmnist': 4,       # Retinal OCT - Multi-Class (4)
    'pneumoniamnist': 2, # Chest X-Ray - Binary-Class (2)
    'retinamnist': 5,    # Fundus Camera - Ordinal Regression (5) - treated as classification
    'breastmnist': 2,    # Breast Ultrasound - Binary-Class (2)
    'bloodmnist': 8,     # Blood Cell Microscope - Multi-Class (8)
    'tissuemnist': 8,    # Kidney Cortex Microscope - Multi-Class (8)
    'organamnist': 11,   # Abdominal CT - Multi-Class (11)
    'organcmnist': 11,   # Abdominal CT - Multi-Class (11)
    'organsmnist': 11,   # Abdominal CT - Multi-Class (11)
}

# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'mobilenet_w1',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet20_cifar100', 'regnetx_600m'
                        ],
                        help='model to be quantized')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        choices=list(CLASSIFICATION_DATASETS.keys()),
                        help='dataset to generate calibration data for')
    parser.add_argument('--image_size',
                        type=int,
                        default=None,
                        help='image size')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--group',
                        type=int,
                        default=1,
                        help='group of generated data')
    parser.add_argument('--beta',
                        type=float,
                        default=1.0,
                        help='beta')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.0,
                        help='gamma')
    parser.add_argument('--save_path_head',
                        type=str,
                        default='',
                        help='save_path_head')
    parser.add_argument('--init_data_path',
                        type=str,
                        default=None,
                        help='path to real images used for initialization')
    parser.add_argument('--synthesis',
                        type=str,
                        default='inversion',
                        choices=['inversion', 'dsv'],
                        help='data synthesis method')

    args = parser.parse_args()
    return args

def convert_state_dict(pretrained_state_dict, new_model):
    """
    Converts a pretrained ResNet-18 state_dict to match the key format of the new model,
    including all 'num_batches_tracked' keys for BatchNorm layers.

    Args:
        pretrained_state_dict (OrderedDict): The state_dict object from the pretrained model.
        new_model (torch.nn.Module): An instance of the new model architecture.

    Returns:
        OrderedDict: The converted state_dict.
    """
    # For debugging: Check if the problematic key exists in the source state_dict
    if 'bn1.num_batches_tracked' not in pretrained_state_dict:
        print("Warning: 'bn1.num_batches_tracked' not found in the source checkpoint!")

    new_state_dict = OrderedDict()
    key_map = {}

    # 1. Initial block (conv1 and bn1)
    key_map.update({
        'conv1.weight': 'features.init_block.conv.conv.weight',
        'bn1.weight': 'features.init_block.conv.bn.weight',
        'bn1.bias': 'features.init_block.conv.bn.bias',
        'bn1.running_mean': 'features.init_block.conv.bn.running_mean',
        'bn1.running_var': 'features.init_block.conv.bn.running_var',
        'bn1.num_batches_tracked': 'features.init_block.conv.bn.num_batches_tracked'  # The missing key
    })

    # 2. ResNet stages (layer1 to layer4)
    for i in range(1, 5):  # Stages 1-4
        for j in range(2):  # Units 1-2 (for ResNet-18)
            # Body convolutions and their batchnorms
            for conv_idx in [1, 2]:
                old_prefix = f'layer{i}.{j}.conv{conv_idx}'
                new_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_prefix}.weight'] = f'{new_prefix}.conv.weight'

                old_bn_prefix = f'layer{i}.{j}.bn{conv_idx}'
                new_bn_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_bn_prefix}.weight'] = f'{new_bn_prefix}.bn.weight'
                key_map[f'{old_bn_prefix}.bias'] = f'{new_bn_prefix}.bn.bias'
                key_map[f'{old_bn_prefix}.running_mean'] = f'{new_bn_prefix}.bn.running_mean'
                key_map[f'{old_bn_prefix}.running_var'] = f'{new_bn_prefix}.bn.running_var'
                key_map[f'{old_bn_prefix}.num_batches_tracked'] = f'{new_bn_prefix}.bn.num_batches_tracked'

            # Downsample (identity) convolution for stages 2, 3, 4
            if i > 1 and j == 0:
                old_ds_prefix = f'layer{i}.{j}.downsample'
                new_ds_prefix = f'features.stage{i}.unit{j+1}.identity_conv'
                key_map[f'{old_ds_prefix}.0.weight'] = f'{new_ds_prefix}.conv.weight'
                key_map[f'{old_ds_prefix}.1.weight'] = f'{new_ds_prefix}.bn.weight'
                key_map[f'{old_ds_prefix}.1.bias'] = f'{new_ds_prefix}.bn.bias'
                key_map[f'{old_ds_prefix}.1.running_mean'] = f'{new_ds_prefix}.bn.running_mean'
                key_map[f'{old_ds_prefix}.1.running_var'] = f'{new_ds_prefix}.bn.running_var'
                key_map[f'{old_ds_prefix}.1.num_batches_tracked'] = f'{new_ds_prefix}.bn.num_batches_tracked'

    # 3. Final fully-connected layer
    key_map.update({
        'fc.weight': 'output.weight',
        'fc.bias': 'output.bias'
    })

    # Populate the new_state_dict using the generated map
    for old_key, new_key in key_map.items():
        if old_key in pretrained_state_dict:
            new_state_dict[new_key] = pretrained_state_dict[old_key]

    return new_state_dict


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    if args.dataset is not None:
        args.dataset = args.dataset.lower()
        # args.model = args.model + '_' + args.dataset
        if args.image_size is not None:
            args.save_path_head = args.save_path_head + '_' + str(args.image_size)
            # pretrained_path = f'../checkpoints/{args.model}_{args.image_size}.pth'
            pretrained_path = f'/home/project/dfq/medmnist/{args.dataset}/{args.model}_{args.image_size}_2.pth'
        else:
            args.save_path_head = args.save_path_head
            pretrained_path = f'../checkpoints/{args.model}_{args.dataset}.pth'

        if not os.path.exists(pretrained_path):
            raise ValueError(f"Pretrained model {pretrained_path} not found")
        
        # Determine the correct number of classes for the dataset
        num_classes = CLASSIFICATION_DATASETS.get(args.dataset, 1000)
        
        if args.image_size == 28:
            model = ResNet18(in_channels=3, num_classes=num_classes, img_size=28)
            print(f'****** ResNet18 model created with {num_classes} classes for {args.dataset} (28x28) ******')
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # if isinstance(checkpoint, dict) and 'net' in checkpoint:
            #     converted_state_dict = convert_state_dict(checkpoint['net'], model)
            # else:
            #     converted_state_dict = convert_state_dict(checkpoint, model)
            model.load_state_dict(checkpoint['net'])
            print(f'****** Pretrained model {pretrained_path} loaded ******')
        else:
            # Create a new model with the correct number of classes
            model = ptcv_get_model(args.model.split('_')[0], pretrained=False, num_classes=num_classes)
            print(f'****** Model created with {num_classes} classes for {args.dataset} ******')
            
            # Load checkpoint and handle different formats
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'net' in checkpoint:
                converted_state_dict = convert_state_dict(checkpoint['net'], model)
            else:
                converted_state_dict = convert_state_dict(checkpoint, model)
        
            model.load_state_dict(converted_state_dict)
            print(f'****** Pretrained model {pretrained_path} loaded ******')

    else:
        model = ptcv_get_model(args.model, pretrained=True)
        print('****** Full precision model loaded ******')

    # # Load validation data
    # test_loader = getTestData(args.dataset,
    #                           batch_size=args.test_batch_size,
    #                           path='/media/disk1/ImageNet2012/',
    #                           for_inception=args.model.startswith('inception'))
    # print('****** Test model! ******')
    # test(model.cuda(), test_loader)
    # Generate distilled data
    DD = DistillData()
    if args.synthesis == 'dsv':
        dataloader = DD.getDistilData_dsv(
            model_name=args.model,
            teacher_model=model.cuda(),
            batch_size=args.batch_size,
            group=args.group,
            beta=args.beta,
            gamma=args.gamma,
            save_path_head=args.save_path_head,
            init_data_path=args.init_data_path
        )
    else:
        dataloader = DD.getDistilData_hardsample(
            model_name=args.model,
            teacher_model=model.cuda(),
            batch_size=args.batch_size,
            group=args.group,
            beta=args.beta,
            gamma=args.gamma,
            save_path_head=args.save_path_head,
            init_data_path=args.init_data_path
        )

    print('****** Data Generated ******')




