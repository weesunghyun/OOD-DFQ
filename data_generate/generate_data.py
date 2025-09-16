import argparse
import json
import math
import os
import pickle
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from pytorchcv.model_provider import get_model as ptcv_get_model

from models import ResNet18


# Classification task datasets from MedMNIST2D
CLASSIFICATION_DATASETS = {
    'pathmnist': 9,
    'dermamnist': 7,
    'octmnist': 4,
    'pneumoniamnist': 2,
    'retinamnist': 5,
    'breastmnist': 2,
    'bloodmnist': 8,
    'tissuemnist': 8,
    'organamnist': 11,
    'organcmnist': 11,
    'organsmnist': 11,
}


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Unified Informativeness-based data curation pipeline for OOD distillation datasets.'
    )
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'mobilenet_w1',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet20_cifar100', 'regnetx_600m'
                        ],
                        help='Teacher model to score the OOD dataset with.')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        choices=list(CLASSIFICATION_DATASETS.keys()),
                        help='Optional dataset identifier when loading a task-specific teacher checkpoint.')
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help='Path to the OOD dataset root (expects ImageFolder structure).')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Directory where curated pickle shards will be stored.')
    parser.add_argument('--file_prefix',
                        type=str,
                        default='unified_curated',
                        help='Base filename prefix for curated data shards (without group suffix).')
    parser.add_argument('--subset_size',
                        type=int,
                        default=500000,
                        help='Number of images sampled for scoring. Use -1 to process the full dataset.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size used during teacher inference.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of workers for the scoring dataloader.')
    parser.add_argument('--num_augmentations',
                        type=int,
                        default=5,
                        help='Number of lightweight augmentations per image when computing metrics.')
    parser.add_argument('--w_sens',
                        type=float,
                        default=0.5,
                        help='Weight for augmentation sensitivity in the unified score.')
    parser.add_argument('--w_pot',
                        type=float,
                        default=0.5,
                        help='Weight for augmentation potential in the unified score.')
    parser.add_argument('--samples_per_class',
                        type=int,
                        default=50,
                        help='Number of curated samples to keep per teacher pseudo-label.')
    parser.add_argument('--max_total_samples',
                        type=int,
                        default=None,
                        help='Optional cap on the total number of curated samples after per-class selection.')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Image resolution fed to the teacher model.')
    parser.add_argument('--num_groups',
                        type=int,
                        default=4,
                        help='Number of pickle shards to create for the curated dataset.')
    parser.add_argument('--metadata_path',
                        type=str,
                        default=None,
                        help='Optional output path for metadata JSON describing curated samples.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed used for subset sampling.')
    parser.add_argument('--save_path_head',
                        type=str,
                        default=None,
                        help='Deprecated alias for --output_dir (kept for backwards compatibility).')
    parser.add_argument('--teacher_checkpoint',
                        type=str,
                        default=None,
                        help='Optional explicit path to a teacher checkpoint when --dataset is provided.')

    args = parser.parse_args()

    if args.output_dir is None:
        if args.save_path_head is not None:
            print('Using deprecated --save_path_head as output directory alias.')
            args.output_dir = args.save_path_head
        else:
            raise ValueError('Please provide --output_dir to specify where curated data should be saved.')

    if args.num_augmentations < 1:
        raise ValueError('--num_augmentations must be >= 1.')

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that also returns the sample path."""

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if image_size >= 64:
        resize_size = int(image_size / 0.875)
    else:
        resize_size = image_size

    base_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    augmentation_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    storage_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    return {
        'base': base_transform,
        'augment': augmentation_transform,
        'storage': storage_transform,
    }


class UnifiedInformativenessCurator:
    def __init__(
        self,
        teacher_model: torch.nn.Module,
        base_transform: transforms.Compose,
        augmentation_transform: transforms.Compose,
        w_sens: float,
        w_pot: float,
        num_augmentations: int,
        device: torch.device,
        image_loader: Callable[[str], object] = default_loader,
    ):
        self.teacher_model = teacher_model.to(device).eval()
        self.base_transform = base_transform
        self.augmentation_transform = augmentation_transform
        self.w_sens = w_sens
        self.w_pot = w_pot
        self.num_augmentations = num_augmentations
        self.device = device
        self.loader = image_loader
        self.eps = 1e-8

    def score_dataset(
        self,
        dataset: datasets.ImageFolder,
        subset_size: Optional[int],
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> List[Dict[str, float]]:
        if subset_size is not None and subset_size > 0 and subset_size < len(dataset):
            generator = torch.Generator()
            generator.manual_seed(seed)
            selected_indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
            working_dataset = Subset(dataset, selected_indices)
            print(f'Scoring a random subset of {len(working_dataset)} / {len(dataset)} images.')
        else:
            working_dataset = dataset
            print(f'Scoring the full dataset with {len(dataset)} images.')

        loader = DataLoader(
            working_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: batch,
        )

        total_batches = math.ceil(len(working_dataset) / batch_size)
        scored_samples: List[Dict[str, float]] = []

        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                images, _, paths = zip(*batch)
                inputs = torch.stack([self.base_transform(img) for img in images]).to(self.device)

                logits = self.teacher_model(inputs)
                probs = F.softmax(logits, dim=1)
                log_probs = torch.log(probs + self.eps)
                pseudo_labels = probs.argmax(dim=1)

                sensitivity_sum = torch.zeros(probs.size(0), device=self.device)
                augmented_probabilities = []

                for _ in range(self.num_augmentations):
                    aug_inputs = torch.stack([self.augmentation_transform(img) for img in images]).to(self.device)
                    aug_logits = self.teacher_model(aug_inputs)
                    aug_probs = F.softmax(aug_logits, dim=1)
                    aug_log_probs = torch.log(aug_probs + self.eps)

                    kl = torch.sum(probs * (log_probs - aug_log_probs), dim=1)
                    sensitivity_sum += kl
                    augmented_probabilities.append(aug_probs)

                sensitivity = sensitivity_sum / float(self.num_augmentations)
                mean_aug_probs = torch.stack(augmented_probabilities, dim=0).mean(dim=0)
                potential = -(mean_aug_probs * torch.log(mean_aug_probs + self.eps)).sum(dim=1)
                scores = self.w_sens * sensitivity + self.w_pot * potential

                sensitivity = sensitivity.cpu()
                potential = potential.cpu()
                scores = scores.cpu()
                pseudo_labels = pseudo_labels.cpu()

                for idx in range(len(paths)):
                    scored_samples.append({
                        'path': paths[idx],
                        'pseudo_label': int(pseudo_labels[idx].item()),
                        'sensitivity': float(sensitivity[idx].item()),
                        'potential': float(potential[idx].item()),
                        'score': float(scores[idx].item()),
                    })

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    print(f'[Scoring] Processed batch {batch_idx + 1}/{total_batches}')

        return scored_samples

    def select_top_samples(
        self,
        scored_samples: List[Dict[str, float]],
        samples_per_class: Optional[int],
        max_total_samples: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        grouped: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        for sample in scored_samples:
            grouped[sample['pseudo_label']].append(sample)

        curated: List[Dict[str, float]] = []
        for pseudo_label, items in grouped.items():
            items.sort(key=lambda x: x['score'], reverse=True)
            limit = len(items)
            if samples_per_class is not None:
                limit = min(limit, samples_per_class)
            curated.extend(items[:limit])
            print(f'Pseudo-label {pseudo_label}: selected {min(limit, len(items))} / {len(items)} samples.')

        if max_total_samples is not None and len(curated) > max_total_samples:
            curated.sort(key=lambda x: x['score'], reverse=True)
            curated = curated[:max_total_samples]
            print(f'Applied global cap of {max_total_samples} curated samples.')

        curated.sort(key=lambda x: x['score'], reverse=True)

        for rank, sample in enumerate(curated):
            sample['rank'] = rank + 1

        return curated

    def save_curated_samples(
        self,
        curated_samples: List[Dict[str, float]],
        storage_transform: transforms.Compose,
        output_dir: str,
        file_prefix: str,
        num_groups: int,
        metadata_path: Optional[str],
    ) -> None:
        if not curated_samples:
            raise ValueError('No curated samples available to save.')

        os.makedirs(output_dir, exist_ok=True)

        total = len(curated_samples)
        group_size = math.ceil(total / float(num_groups))
        data_prefix = os.path.join(output_dir, f'{file_prefix}_group')
        label_prefix = os.path.join(output_dir, f'{file_prefix}_labels_group')

        saved_groups = 0
        start = 0
        metadata_entries: List[Dict[str, float]] = []

        for group_idx in range(num_groups):
            end = min(start + group_size, total)
            group_samples = curated_samples[start:end]
            if not group_samples:
                break

            tensors = []
            labels = []
            for sample in group_samples:
                image = self.loader(sample['path'])
                tensor = storage_transform(image)
                tensors.append(tensor.numpy())
                labels.append(sample['pseudo_label'])

                entry = dict(sample)
                entry['group'] = group_idx + 1
                metadata_entries.append(entry)

            data_array = np.stack(tensors, axis=0)
            label_array = np.array(labels, dtype=np.int64)

            data_file = f'{data_prefix}{group_idx + 1}.pickle'
            label_file = f'{label_prefix}{group_idx + 1}.pickle'

            with open(data_file, 'wb') as fp:
                pickle.dump([data_array], fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open(label_file, 'wb') as fp:
                pickle.dump([label_array], fp, protocol=pickle.HIGHEST_PROTOCOL)

            print(f'Saved group {group_idx + 1}: {len(group_samples)} samples -> {data_file}')
            saved_groups += 1
            start = end

        if metadata_path is None:
            metadata_path = os.path.join(output_dir, f'{file_prefix}_metadata.json')

        metadata = {
            'total_samples': total,
            'saved_groups': saved_groups,
            'num_augmentations': self.num_augmentations,
            'weights': {'sensitivity': self.w_sens, 'potential': self.w_pot},
            'entries': metadata_entries,
        }

        with open(metadata_path, 'w') as fp:
            json.dump(metadata, fp, indent=2)

        print(f'Metadata written to {metadata_path}')


def convert_state_dict(pretrained_state_dict, new_model):
    """Converts pretrained ResNet-18 state dict to the pytorchcv format."""
    new_state_dict = {}

    key_map = {
        'conv1.weight': 'features.init_block.conv.conv.weight',
        'bn1.weight': 'features.init_block.conv.bn.weight',
        'bn1.bias': 'features.init_block.conv.bn.bias',
        'bn1.running_mean': 'features.init_block.conv.bn.running_mean',
        'bn1.running_var': 'features.init_block.conv.bn.running_var',
        'bn1.num_batches_tracked': 'features.init_block.conv.bn.num_batches_tracked',
        'fc.weight': 'output.weight',
        'fc.bias': 'output.bias',
    }

    for i in range(1, 5):
        for j in range(2):
            for conv_idx in [1, 2]:
                old_prefix = f'layer{i}.{j}.conv{conv_idx}'
                new_prefix = f'features.stage{i}.unit{j + 1}.body.conv{conv_idx}'
                key_map[f'{old_prefix}.weight'] = f'{new_prefix}.conv.weight'

                old_bn_prefix = f'layer{i}.{j}.bn{conv_idx}'
                new_bn_prefix = f'features.stage{i}.unit{j + 1}.body.conv{conv_idx}'
                key_map[f'{old_bn_prefix}.weight'] = f'{new_bn_prefix}.bn.weight'
                key_map[f'{old_bn_prefix}.bias'] = f'{new_bn_prefix}.bn.bias'
                key_map[f'{old_bn_prefix}.running_mean'] = f'{new_bn_prefix}.bn.running_mean'
                key_map[f'{old_bn_prefix}.running_var'] = f'{new_bn_prefix}.bn.running_var'
                key_map[f'{old_bn_prefix}.num_batches_tracked'] = f'{new_bn_prefix}.bn.num_batches_tracked'

            if i > 1 and j == 0:
                old_ds_prefix = f'layer{i}.{j}.downsample'
                new_ds_prefix = f'features.stage{i}.unit{j + 1}.identity_conv'
                key_map[f'{old_ds_prefix}.0.weight'] = f'{new_ds_prefix}.conv.weight'
                key_map[f'{old_ds_prefix}.1.weight'] = f'{new_ds_prefix}.bn.weight'
                key_map[f'{old_ds_prefix}.1.bias'] = f'{new_ds_prefix}.bn.bias'
                key_map[f'{old_ds_prefix}.1.running_mean'] = f'{new_ds_prefix}.bn.running_mean'
                key_map[f'{old_ds_prefix}.1.running_var'] = f'{new_ds_prefix}.bn.running_var'
                key_map[f'{old_ds_prefix}.1.num_batches_tracked'] = f'{new_ds_prefix}.bn.num_batches_tracked'

    for old_key, new_key in key_map.items():
        if old_key in pretrained_state_dict:
            new_state_dict[new_key] = pretrained_state_dict[old_key]

    return new_state_dict


def load_teacher_model(args) -> torch.nn.Module:
    if args.dataset is not None:
        dataset_key = args.dataset.lower()
        num_classes = CLASSIFICATION_DATASETS.get(dataset_key, 1000)

        if args.teacher_checkpoint is not None:
            checkpoint_path = args.teacher_checkpoint
        else:
            if args.image_size == 28:
                checkpoint_path = f'/home/project/dfq/medmnist/{dataset_key}/{args.model}_{args.image_size}_2.pth'
            else:
                checkpoint_path = f'../checkpoints/{args.model}_{dataset_key}.pth'

        if not os.path.exists(checkpoint_path):
            raise ValueError(f'Teacher checkpoint not found at {checkpoint_path}. Please provide --teacher_checkpoint.')

        if args.image_size == 28:
            model = ResNet18(in_channels=3, num_classes=num_classes, img_size=28)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        else:
            model_name = args.model.split('_')[0]
            model = ptcv_get_model(model_name, pretrained=False, num_classes=num_classes)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
            converted_state_dict = convert_state_dict(state_dict, model)
            model.load_state_dict(converted_state_dict)

        print(f'Loaded teacher checkpoint from {checkpoint_path}')
        return model

    if args.teacher_checkpoint is not None:
        model = ptcv_get_model(args.model, pretrained=False)
        checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
        state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f'Loaded teacher checkpoint from {args.teacher_checkpoint}')
        return model

    print('Loading ImageNet-pretrained teacher from pytorchcv.')
    teacher = ptcv_get_model(args.model, pretrained=True)
    return teacher


def summarize_scores(scored_samples: List[Dict[str, float]]) -> Dict[str, float]:
    sensitivities = np.array([sample['sensitivity'] for sample in scored_samples])
    potentials = np.array([sample['potential'] for sample in scored_samples])
    scores = np.array([sample['score'] for sample in scored_samples])
    return {
        'sensitivity_mean': float(sensitivities.mean()),
        'sensitivity_std': float(sensitivities.std()),
        'potential_mean': float(potentials.mean()),
        'potential_std': float(potentials.std()),
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std()),
    }


def main():
    args = arg_parse()
    set_seed(args.seed)

    if not os.path.isdir(args.dataset_path):
        raise ValueError(f'Dataset path {args.dataset_path} does not exist or is not a directory.')

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    teacher_model = load_teacher_model(args)

    transforms_dict = build_transforms(args.image_size)
    dataset = ImageFolderWithPaths(root=args.dataset_path, transform=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curator = UnifiedInformativenessCurator(
        teacher_model=teacher_model,
        base_transform=transforms_dict['base'],
        augmentation_transform=transforms_dict['augment'],
        w_sens=args.w_sens,
        w_pot=args.w_pot,
        num_augmentations=args.num_augmentations,
        device=device,
        image_loader=dataset.loader,
    )

    subset_size = None if args.subset_size in (-1, None) else args.subset_size
    scored_samples = curator.score_dataset(
        dataset=dataset,
        subset_size=subset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    summary = summarize_scores(scored_samples)
    print('Scoring summary:')
    for key, value in summary.items():
        print(f'  {key}: {value:.6f}')

    curated_samples = curator.select_top_samples(
        scored_samples=scored_samples,
        samples_per_class=args.samples_per_class,
        max_total_samples=args.max_total_samples,
    )

    print(f'Total curated samples: {len(curated_samples)}')

    curator.save_curated_samples(
        curated_samples=curated_samples,
        storage_transform=transforms_dict['storage'],
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        num_groups=args.num_groups,
        metadata_path=args.metadata_path,
    )


if __name__ == '__main__':
    main()

