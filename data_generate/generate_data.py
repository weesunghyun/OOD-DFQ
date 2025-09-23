import argparse
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from heapq import heappush, heappushpop, nlargest
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:  # pragma: no cover - optional dependency
    MiniBatchKMeans = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def compute_even_class_targets(total: int, num_classes: int) -> Dict[int, int]:
    if num_classes <= 0:
        raise ValueError('Number of classes must be positive to distribute targets evenly.')

    base = total // num_classes
    remainder = total % num_classes

    targets = {label: base for label in range(num_classes)}

    for label in range(remainder):
        targets[label] += 1

    return targets


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
                        default=-1,
                        help='Number of images sampled for initial pseudo-labeling. Use -1 to process the full dataset.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Batch size used during teacher inference.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=16,
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
    parser.add_argument('--candidate_pool_per_class',
                        type=int,
                        default=100,
                        help='Number of pseudo-labeled candidates to retain per class before augmentation scoring.')
    parser.add_argument('--total_candidate_pool',
                        type=int,
                        default=None,
                        help='Total number of pseudo-labeled candidates to retain across all classes before augmentation scoring.')
    parser.add_argument('--adaptive_min_samples_per_class',
                        type=int,
                        default=0,
                        help='Minimum number of pseudo-labeled samples to retain per class during the adaptive search stage.')
    parser.add_argument('--adaptive_sample_chunk_size',
                        type=int,
                        default=50000,
                        help='Number of images processed between adaptive search status reports. Set <= 0 to disable chunk reporting.')
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
    parser.add_argument('--total_samples',
                        type=int,
                        default=None,
                        help='Total number of curated samples to keep across all classes after augmentation scoring.')
    parser.add_argument('--sampling_strategy',
                        type=str,
                        default='pseudo_label',
                        choices=['pseudo_label', 'feature_diversity', 'meta_label'],
                        help='Sampling strategy used to construct the curated dataset.')
    parser.add_argument('--feature_candidate_pool_size',
                        type=int,
                        default=50000,
                        help='Top-K samples retained before clustering when using feature-diversity sampling. '
                             'Use <= 0 to disable pre-filtering.')
    parser.add_argument('--feature_cluster_count',
                        type=int,
                        default=100,
                        help='Number of Mini-Batch K-Means clusters for feature-diversity sampling.')
    parser.add_argument('--feature_samples_per_cluster',
                        type=int,
                        default=None,
                        help='Number of samples retained per cluster for feature-diversity sampling. '
                             'If omitted, the total target is evenly distributed across clusters.')
    parser.add_argument('--meta_top_n',
                        type=int,
                        default=3,
                        help='Number of top teacher predictions used to form meta-labels.')
    parser.add_argument('--meta_label_top_k',
                        type=int,
                        default=50,
                        help='Number of most frequent meta-label groups kept during meta-label sampling.')
    parser.add_argument('--meta_samples_per_group',
                        type=int,
                        default=None,
                        help='Optional fixed number of samples per meta-label group when using meta-label sampling. '
                             'If omitted, the total target is divided evenly across groups.')
    parser.add_argument('--meta_include_others',
                        action='store_true',
                        help='Include the aggregated "others" meta-label bucket during reallocation when using '
                             'meta-label sampling.')
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


class CandidateDataset(Dataset):
    """Dataset wrapper around candidate samples identified by pseudo-labeling."""

    def __init__(self, candidates: List[Dict[str, Any]], image_loader: Callable[[str], object]):
        self.candidates = candidates
        self.loader = image_loader

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int):
        entry = self.candidates[index]
        image = self.loader(entry['path'])
        pseudo_label = entry['pseudo_label']
        path = entry['path']
        return image, pseudo_label, path


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean()
        variance = max((self.total_sq / self.count) - mean * mean, 0.0)
        return math.sqrt(variance)


class ScoreSummaryAccumulator:
    def __init__(self) -> None:
        self.metrics: Dict[str, RunningStats] = {
            'sensitivity': RunningStats(),
            'potential': RunningStats(),
            'score': RunningStats(),
        }

    def update(self, sample: Dict[str, float]) -> None:
        for key, stats in self.metrics.items():
            stats.update(float(sample[key]))

    def to_dict(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, stats in self.metrics.items():
            summary[f'{name}_mean'] = stats.mean()
            summary[f'{name}_std'] = stats.std()
        return summary

    @property
    def count(self) -> int:
        # All metrics are updated together, so reuse the score metric's count.
        return self.metrics['score'].count


class PerClassSampleSelector:
    def __init__(
        self,
        default_limit: Optional[int],
        per_class_limits: Optional[Dict[int, int]] = None,
        global_cap: Optional[int] = None,
    ) -> None:
        self.default_limit = default_limit
        self.per_class_limits = per_class_limits or {}
        self.global_cap = global_cap
        self.entries_by_class: Dict[int, List[Tuple[float, int, Dict[str, Any]]]] = defaultdict(list)
        self.unlimited_entries: List[Tuple[float, int, Dict[str, Any]]] = []
        self.counter = 0

    def add(self, sample: Dict[str, Any]) -> None:
        label = sample['pseudo_label']
        limit = self.per_class_limits.get(label, self.default_limit)

        if limit is not None and limit <= 0:
            return

        entry = (sample['score'], self.counter, sample)
        self.counter += 1

        if limit is None:
            self.unlimited_entries.append(entry)
        else:
            self.entries_by_class[label].append(entry)

    def finalize(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        reallocation_pool: List[Tuple[float, int, Dict[str, Any]]] = []

        for label, entries in self.entries_by_class.items():
            limit = self.per_class_limits.get(label, self.default_limit)

            if limit is None:
                reallocation_pool.extend(entries)
                continue

            if limit <= 0:
                continue

            sorted_entries = sorted(entries, key=lambda x: (-x[0], x[1]))
            selected = sorted_entries[:limit]
            results.extend(item[2] for item in selected)
            reallocation_pool.extend(sorted_entries[limit:])

        unlimited_sorted = sorted(self.unlimited_entries, key=lambda x: (-x[0], x[1]))

        target_total: Optional[int] = None
        effective_global_cap = (
            self.global_cap if self.global_cap is None or self.global_cap >= 0 else None
        )
        if self.per_class_limits:
            target_total = sum(
                max(limit, 0)
                for limit in self.per_class_limits.values()
                if limit is not None
            )

        if (
            target_total is None
            and self.default_limit is not None
            and self.default_limit > 0
        ):
            observed_classes = len(self.entries_by_class)
            if observed_classes > 0:
                target_total = self.default_limit * observed_classes

        if target_total is None and effective_global_cap is not None:
            target_total = effective_global_cap
        elif target_total is not None and effective_global_cap is not None:
            target_total = min(target_total, effective_global_cap)

        if target_total is None:
            results.extend(item[2] for item in unlimited_sorted)
        else:
            reallocation_pool.extend(unlimited_sorted)
            if len(results) < target_total and reallocation_pool:
                needed = target_total - len(results)
                reallocation_pool.sort(key=lambda x: (-x[0], x[1]))
                additions = reallocation_pool[:needed]
                if additions:
                    results.extend(item[2] for item in additions)
                    print(
                        f'Reallocated {len(additions)} sample(s) across pseudo-labels '
                        f'to satisfy the total target of {target_total}.'
                    )

            if len(results) < target_total:
                print(
                    f'Warning: Only collected {len(results)} curated samples '
                    f'out of the desired total of {target_total} after reallocation.'
                )

        results.sort(key=lambda x: x['score'], reverse=True)

        trimmed = False
        if self.global_cap is not None and self.global_cap >= 0 and len(results) > self.global_cap:
            results = results[:self.global_cap]
            trimmed = True

        for rank, sample in enumerate(results, 1):
            sample['rank'] = rank

        self.entries_by_class.clear()
        self.unlimited_entries.clear()

        if trimmed:
            print(f'Applied global cap of {self.global_cap} curated samples.')

        return results


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

    def build_candidate_pool(
        self,
        dataset: datasets.ImageFolder,
        candidate_pool_per_class: Optional[int],
        candidate_pool_targets: Optional[Dict[int, int]],
        batch_size: int,
        num_workers: int,
        subset_size: Optional[int],
        seed: int,
        num_classes: Optional[int] = None,
        adaptive_min_samples_per_class: int = 0,
        adaptive_sample_chunk_size: int = 0,
    ) -> List[Dict[str, Any]]:
        if subset_size is not None and subset_size > 0 and subset_size < len(dataset):
            generator = torch.Generator()
            generator.manual_seed(seed)
            selected_indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
            working_dataset = Subset(dataset, selected_indices)
            print(f'Pseudo-labeling a random subset of {len(working_dataset)} / {len(dataset)} images.')
        else:
            working_dataset = dataset
            print(f'Pseudo-labeling the full dataset with {len(dataset)} images.')

        loader = DataLoader(
            working_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: batch,
        )

        total_batches = math.ceil(len(working_dataset) / batch_size)

        default_candidate_limit = None
        if candidate_pool_per_class is not None and candidate_pool_per_class > 0:
            default_candidate_limit = candidate_pool_per_class

        candidate_limits = candidate_pool_targets or {}

        candidate_heaps: Dict[int, List[Tuple[float, int, Dict[str, Any]]]] = defaultdict(list)
        candidate_lists: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        class_counts: Dict[int, int] = defaultdict(int)
        kept_counts: Dict[int, int] = defaultdict(int)
        applied_limits: Dict[int, Optional[int]] = {}
        candidate_counter = 0

        adaptive_target = adaptive_min_samples_per_class if adaptive_min_samples_per_class and adaptive_min_samples_per_class > 0 else None
        chunk_size = None
        if adaptive_target is not None and adaptive_sample_chunk_size and adaptive_sample_chunk_size > 0:
            chunk_size = adaptive_sample_chunk_size
        processed_samples = 0
        next_report = chunk_size if chunk_size is not None else None

        pending_labels: Set[int] = set()
        pending_limit_labels: Set[int] = set()
        dynamic_pending = False
        if adaptive_target is not None:
            if num_classes is not None and num_classes > 0:
                pending_labels = set(range(num_classes))
            else:
                dynamic_pending = True
                print(
                    'Warning: adaptive_min_samples_per_class was provided without a known number of teacher classes. '
                    'Adaptive balancing will track labels as they are observed.'
                )

        limit_adjustment_notified: Set[int] = set()
        zero_limit_notified: Set[int] = set()

        with torch.inference_mode():
            completion_reached = False
            for batch in tqdm(loader, desc="Pseudo-labeling batches", total=total_batches):
                images, _, paths = zip(*batch)
                inputs = torch.stack([self.base_transform(img) for img in images]).to(self.device)

                logits = self.teacher_model(inputs)
                probs = F.softmax(logits, dim=1)
                max_probs, pseudo_labels = probs.max(dim=1)

                for idx in range(len(paths)):
                    label = int(pseudo_labels[idx].item())
                    confidence = float(max_probs[idx].item())

                    class_counts[label] += 1
                    candidate_counter += 1

                    if adaptive_target is not None and dynamic_pending and label not in pending_labels:
                        pending_labels.add(label)

                    limit = candidate_limits.get(label, default_candidate_limit)
                    effective_limit = limit

                    if adaptive_target is not None:
                        target = adaptive_target
                        if effective_limit is None:
                            effective_limit = target
                        elif effective_limit < target:
                            if label not in limit_adjustment_notified:
                                print(
                                    f'Increasing candidate limit for pseudo-label {label} '
                                    f'from {effective_limit} to satisfy adaptive minimum of {target}.'
                                )
                                limit_adjustment_notified.add(label)
                            effective_limit = target

                    if effective_limit is not None and effective_limit <= 0:
                        applied_limits[label] = effective_limit
                        if adaptive_target is not None and label in pending_labels and label not in zero_limit_notified:
                            print(
                                f'Warning: pseudo-label {label} has a non-positive candidate limit; '
                                'it will be excluded from adaptive balancing.'
                            )
                            zero_limit_notified.add(label)
                            pending_labels.discard(label)

                        pending_limit_labels.discard(label)

                        processed_samples += 1
                        if chunk_size is not None and next_report is not None and processed_samples >= next_report:
                            remaining = sorted(pending_labels)
                            remaining_display = remaining if remaining else 'None'
                            print(
                                f'[Adaptive Search] Processed {processed_samples} samples. '
                                f'Remaining classes below target: {remaining_display}. '
                                f'Current pool sizes: {sum(kept_counts.values())}'
                            )
                            next_report += chunk_size

                        continue

                    applied_limits[label] = effective_limit

                    entry: Dict[str, Any] = {
                        'path': paths[idx],
                        'pseudo_label': label,
                        'candidate_confidence': confidence,
                    }

                    stored = False
                    if effective_limit is None:
                        candidate_lists[label].append(entry)
                        kept_counts[label] = len(candidate_lists[label])
                        stored = True
                    else:
                        heap_entry = (confidence, candidate_counter, entry)
                        heap = candidate_heaps[label]
                        if len(heap) < effective_limit:
                            heappush(heap, heap_entry)
                            stored = True
                        else:
                            popped = heappushpop(heap, heap_entry)
                            stored = popped[2] is not entry
                        kept_counts[label] = len(heap)

                    if effective_limit is not None:
                        if kept_counts[label] >= effective_limit:
                            pending_limit_labels.discard(label)
                        else:
                            pending_limit_labels.add(label)
                    else:
                        pending_limit_labels.discard(label)

                    if adaptive_target is not None and stored:
                        if kept_counts[label] >= adaptive_target and label in pending_labels:
                            pending_labels.discard(label)

                    processed_samples += 1
                    if chunk_size is not None and next_report is not None and processed_samples >= next_report:
                        remaining = sorted(pending_labels)
                        remaining_display = remaining if remaining else 'None'
                        print(
                            f'[Adaptive Search] Processed {processed_samples} samples. '
                            f'Remaining classes below target: {remaining_display}. '
                            f'Current pool sizes: {sum(kept_counts.values())}'
                        )
                        next_report += chunk_size

                    if adaptive_target is not None and not pending_labels and not pending_limit_labels:
                        completion_reached = True
                        break

                if completion_reached:
                    break

        if adaptive_target is not None:
            if pending_labels:
                print('Warning: Unable to satisfy adaptive minimum for the following pseudo-labels:')
                for label in sorted(pending_labels):
                    print(
                        f'  class {label}: gathered {kept_counts.get(label, 0)} / {adaptive_target} candidates '
                        f'(total seen: {class_counts.get(label, 0)})'
                    )
            else:
                print(
                    'Adaptive search satisfied the minimum candidate requirement for all pseudo-labels '
                    f'({adaptive_target} per class).'
                )

        candidate_pool: List[Dict[str, Any]] = []
        total_candidates = 0

        all_labels = set(class_counts.keys())
        all_labels.update(candidate_limits.keys())
        all_labels.update(applied_limits.keys())

        for label in sorted(all_labels):
            limit = applied_limits.get(label)
            if limit is None:
                limit = candidate_limits.get(label, default_candidate_limit)

            if limit is None:
                label_candidates = candidate_lists[label]
            else:
                heap = candidate_heaps[label]
                top_items = nlargest(len(heap), heap)
                label_candidates = [item[2] for item in top_items]

            label_candidates.sort(key=lambda x: x['candidate_confidence'], reverse=True)

            for rank, entry in enumerate(label_candidates, 1):
                entry['candidate_rank'] = rank
                candidate_pool.append(entry)

            total_candidates += len(label_candidates)
            print(
                f'Pseudo-label {label}: total {class_counts.get(label, 0)} samples. '
                f'Candidate pool size: {len(label_candidates)}'
            )

        print(f'Total candidate samples: {total_candidates}')

        return candidate_pool

    def score_dataset_iter(
        self,
        dataset: Dataset,
        subset_size: Optional[int],
        batch_size: int,
        num_workers: int,
        seed: int,
        top_n: Optional[int] = None,
    ) -> Iterator[Dict[str, float]]:
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
        inv_num_augmentations = 1.0 / float(self.num_augmentations)

        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Scoring batches", total=total_batches)):
                images, _, paths = zip(*batch)
                inputs = torch.stack([self.base_transform(img) for img in images]).to(self.device)

                logits = self.teacher_model(inputs)
                probs = F.softmax(logits, dim=1)
                log_probs = torch.log(probs + self.eps)
                pseudo_labels = probs.argmax(dim=1)

                sensitivity_sum = torch.zeros(probs.size(0), device=self.device)
                mean_aug_prob_sum = torch.zeros_like(probs)

                for _ in range(self.num_augmentations):
                    aug_inputs = torch.stack([self.augmentation_transform(img) for img in images]).to(self.device)
                    aug_logits = self.teacher_model(aug_inputs)
                    aug_probs = F.softmax(aug_logits, dim=1)
                    aug_log_probs = torch.log(aug_probs + self.eps)

                    kl = torch.sum(probs * (log_probs - aug_log_probs), dim=1)
                    sensitivity_sum += kl
                    mean_aug_prob_sum += aug_probs

                sensitivity = sensitivity_sum * inv_num_augmentations
                mean_aug_probs = mean_aug_prob_sum * inv_num_augmentations
                potential = -(mean_aug_probs * torch.log(mean_aug_probs + self.eps)).sum(dim=1)
                scores = self.w_sens * sensitivity + self.w_pot * potential

                sensitivity = sensitivity.cpu()
                potential = potential.cpu()
                scores = scores.cpu()
                pseudo_labels = pseudo_labels.cpu()

                sorted_top_indices: Optional[torch.Tensor] = None
                if top_n is not None and top_n > 0:
                    top_k = min(top_n, probs.size(1))
                    top_indices = torch.topk(probs, k=top_k, dim=1).indices
                    sorted_top_indices = torch.sort(top_indices, dim=1).values.cpu()

                for idx in range(len(paths)):
                    sample: Dict[str, Any] = {
                        'path': paths[idx],
                        'pseudo_label': int(pseudo_labels[idx].item()),
                        'sensitivity': float(sensitivity[idx].item()),
                        'potential': float(potential[idx].item()),
                        'score': float(scores[idx].item()),
                    }

                    if sorted_top_indices is not None:
                        indices_list = [int(v) for v in sorted_top_indices[idx].tolist()]
                        sample['top_n_indices'] = indices_list
                        sample['meta_label'] = tuple(indices_list)

                    yield sample

                # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                #     print(f'[Scoring] Processed batch {batch_idx + 1}/{total_batches]')

    def score_dataset(
        self,
        dataset: Dataset,
        subset_size: Optional[int],
        batch_size: int,
        num_workers: int,
        seed: int,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        return list(self.score_dataset_iter(
            dataset=dataset,
            subset_size=subset_size,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            top_n=top_n,
        ))

    def extract_features(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
    ) -> np.ndarray:
        """Extracts penultimate features for the provided dataset."""

        linear_layer: Optional[nn.Module] = None
        for module in self.teacher_model.modules():
            if isinstance(module, nn.Linear):
                linear_layer = module

        if linear_layer is None:
            raise ValueError(
                'Unable to locate a Linear layer in the teacher model for feature extraction.'
            )

        captured_batches: List[np.ndarray] = []
        dataset_length = len(dataset)
        features_array: Optional[np.ndarray] = None
        write_index = 0

        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not inputs:
                return
            features = inputs[0]
            if not isinstance(features, torch.Tensor):
                return
            captured_batches.append(features.detach().cpu().numpy())

        handle = linear_layer.register_forward_hook(hook)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: batch,
        )

        try:
            with torch.inference_mode():
                for batch in tqdm(loader, desc="Extracting features"):
                    captured_batches.clear()
                    images, _, _ = zip(*batch)
                    inputs = torch.stack([
                        self.base_transform(img) for img in images
                    ]).to(self.device)
                    _ = self.teacher_model(inputs)

                    if not captured_batches:
                        raise RuntimeError('Feature hook did not capture outputs for the batch.')

                    # Use the first captured batch (there should be only one per forward)
                    batch_features = captured_batches[0]
                    if batch_features.shape[0] != len(images):
                        raise RuntimeError(
                            'Mismatch between captured features and batch size during feature extraction.'
                        )

                    if features_array is None:
                        feature_shape = batch_features.shape[1:]
                        features_array = np.empty(
                            (dataset_length,) + feature_shape,
                            dtype=np.float32,
                        )

                    end_index = write_index + batch_features.shape[0]
                    if end_index > features_array.shape[0]:
                        raise RuntimeError('Calculated feature slice exceeds preallocated array bounds.')

                    features_array[write_index:end_index] = batch_features.astype(np.float32, copy=False)
                    write_index = end_index

        finally:
            handle.remove()

        if features_array is None:
            return np.empty((0, 0), dtype=np.float32)

        return features_array[:write_index]

    def select_top_samples(
        self,
        scored_samples: List[Dict[str, float]],
        samples_per_class: Optional[int],
        max_total_samples: Optional[int] = None,
        per_class_limits: Optional[Dict[int, int]] = None,
    ) -> List[Dict[str, float]]:
        grouped: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        for sample in scored_samples:
            grouped[sample['pseudo_label']].append(sample)

        curated: List[Dict[str, float]] = []
        all_labels = set(grouped.keys())
        if per_class_limits is not None:
            all_labels.update(per_class_limits.keys())

        for pseudo_label in sorted(all_labels):
            items = grouped.get(pseudo_label, [])
            if items:
                items.sort(key=lambda x: x['score'], reverse=True)

            available = len(items)
            limit: Optional[int] = None
            if per_class_limits is not None and pseudo_label in per_class_limits:
                limit = per_class_limits[pseudo_label]
            elif samples_per_class is not None:
                limit = samples_per_class

            take = available if limit is None else min(available, limit)

            curated.extend(items[:take])
            print(f'Pseudo-label {pseudo_label}: selected {take} / {available} samples.')

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


def run_pseudo_label_sampling(
    curator: "UnifiedInformativenessCurator",
    dataset: ImageFolderWithPaths,
    args: argparse.Namespace,
    subset_size: Optional[int],
    num_teacher_classes: int,
) -> List[Dict[str, Any]]:
    candidate_pool_per_class = args.candidate_pool_per_class
    candidate_pool_targets = None
    if args.total_candidate_pool is not None:
        if args.total_candidate_pool <= 0:
            raise ValueError('--total_candidate_pool must be positive when provided.')
        candidate_pool_targets = compute_even_class_targets(
            args.total_candidate_pool, num_teacher_classes
        )
        candidate_pool_per_class = None
        print(
            f'Distributing total candidate pool of {args.total_candidate_pool} '
            f'across {num_teacher_classes} pseudo-labels:'
        )
        for label in range(num_teacher_classes):
            print(f'  class {label}: {candidate_pool_targets.get(label, 0)}')

    samples_per_class = args.samples_per_class
    per_class_sample_limits: Optional[Dict[int, int]] = None
    max_total_samples = args.max_total_samples
    if args.total_samples is not None:
        if args.total_samples <= 0:
            raise ValueError('--total_samples must be positive when provided.')
        per_class_sample_limits = compute_even_class_targets(
            args.total_samples, num_teacher_classes
        )
        samples_per_class = None
        print(
            f'Distributing total curated sample target of {args.total_samples} '
            f'across {num_teacher_classes} pseudo-labels:'
        )
        for label in range(num_teacher_classes):
            print(f'  class {label}: {per_class_sample_limits.get(label, 0)}')

        if max_total_samples is None:
            max_total_samples = args.total_samples
        else:
            new_cap = min(max_total_samples, args.total_samples)
            if new_cap < max_total_samples:
                print(
                    f'Warning: reducing total cap from {max_total_samples} to {new_cap} '
                    f'to satisfy --total_samples={args.total_samples}.'
                )
            max_total_samples = new_cap

    candidate_pool = curator.build_candidate_pool(
        dataset=dataset,
        candidate_pool_per_class=candidate_pool_per_class,
        candidate_pool_targets=candidate_pool_targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=subset_size,
        seed=args.seed,
        num_classes=num_teacher_classes,
        adaptive_min_samples_per_class=args.adaptive_min_samples_per_class,
        adaptive_sample_chunk_size=args.adaptive_sample_chunk_size,
    )

    if not candidate_pool:
        raise ValueError('Candidate pool is empty after pseudo-labeling.')

    candidate_lookup = {entry['path']: entry for entry in candidate_pool}
    candidate_dataset = CandidateDataset(candidate_pool, image_loader=dataset.loader)

    selector = PerClassSampleSelector(
        default_limit=samples_per_class,
        per_class_limits=per_class_sample_limits,
        global_cap=max_total_samples,
    )
    summary_accumulator = ScoreSummaryAccumulator()
    class_totals: Dict[int, int] = defaultdict(int)
    mismatches = 0

    for sample in curator.score_dataset_iter(
        dataset=candidate_dataset,
        subset_size=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    ):
        summary_accumulator.update(sample)
        metadata = candidate_lookup.get(sample['path'])
        label_for_selection = sample['pseudo_label']

        if metadata is not None:
            predicted_label = label_for_selection
            sample['predicted_pseudo_label'] = predicted_label
            sample['initial_pseudo_label'] = metadata['pseudo_label']
            sample['candidate_confidence'] = metadata['candidate_confidence']
            sample['candidate_rank'] = metadata['candidate_rank']
            label_for_selection = metadata['pseudo_label']
            sample['pseudo_label'] = label_for_selection

            if predicted_label != label_for_selection:
                mismatches += 1
        else:
            sample['pseudo_label'] = label_for_selection

        class_totals[label_for_selection] += 1
        selector.add(sample)

    if summary_accumulator.count == 0:
        raise ValueError('No samples were scored. Please check the candidate pool and parameters.')

    if mismatches > 0:
        print(f'Warning: {mismatches} candidate(s) changed pseudo-label during augmentation scoring.')

    summary = summary_accumulator.to_dict()
    print('Scoring summary:')
    for key, value in summary.items():
        print(f'  {key}: {value:.6f}')

    curated_samples = selector.finalize()

    selected_counts: Dict[int, int] = defaultdict(int)
    for sample in curated_samples:
        selected_counts[sample['pseudo_label']] += 1

    all_report_labels = set(class_totals.keys())
    if per_class_sample_limits is not None:
        all_report_labels.update(per_class_sample_limits.keys())

    for label in sorted(all_report_labels):
        available = class_totals.get(label, 0)
        chosen = selected_counts.get(label, 0)
        print(f'Pseudo-label {label}: selected {chosen} / {available} samples.')

    print(f'Total curated samples: {len(curated_samples)}')

    return curated_samples


def run_feature_diversity_sampling(
    curator: "UnifiedInformativenessCurator",
    dataset: ImageFolderWithPaths,
    args: argparse.Namespace,
    subset_size: Optional[int],
) -> List[Dict[str, Any]]:
    summary_accumulator = ScoreSummaryAccumulator()
    candidate_pool_size = args.feature_candidate_pool_size
    collect_all = candidate_pool_size is None or candidate_pool_size <= 0

    scored_samples: List[Dict[str, Any]] = []
    topk_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    total_scored = 0

    for sample_index, sample in enumerate(
        curator.score_dataset_iter(
            dataset=dataset,
            subset_size=subset_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    ):
        summary_accumulator.update(sample)
        total_scored += 1

        if collect_all:
            scored_samples.append(dict(sample))
            continue

        score = sample['score']
        if len(topk_heap) < candidate_pool_size:
            heappush(topk_heap, (score, sample_index, dict(sample)))
        elif score > topk_heap[0][0]:
            heappushpop(topk_heap, (score, sample_index, dict(sample)))

    if total_scored == 0:
        raise ValueError('No samples were scored when performing feature-diversity sampling.')

    summary = summary_accumulator.to_dict()
    print('Scoring summary:')
    for key, value in summary.items():
        print(f'  {key}: {value:.6f}')

    if collect_all:
        candidate_pool = scored_samples
    else:
        candidate_pool = [
            entry[2]
            for entry in sorted(topk_heap, key=lambda item: (-item[0], item[1]))
        ]

    print(f'Candidate pool size after informativeness filtering: {len(candidate_pool)}')

    if MiniBatchKMeans is None:
        raise ImportError(
            'scikit-learn is required for feature-diversity sampling. '
            'Please install it to use this strategy.'
        )

    candidate_dataset = CandidateDataset(candidate_pool, image_loader=dataset.loader)
    features = curator.extract_features(
        dataset=candidate_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if features.shape[0] != len(candidate_pool):
        raise RuntimeError('Feature extraction count mismatch with candidate pool.')

    num_clusters = min(args.feature_cluster_count, len(candidate_pool))
    if num_clusters <= 0:
        raise ValueError('Number of clusters for feature-diversity sampling must be positive.')

    mbatch = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=min(args.batch_size, len(candidate_pool)),
        random_state=args.seed,
    )
    cluster_ids = mbatch.fit_predict(features)

    clusters: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry, cluster_id in zip(candidate_pool, cluster_ids):
        sample = dict(entry)
        sample['feature_cluster'] = int(cluster_id)
        clusters[int(cluster_id)].append(sample)

    total_target: Optional[int] = None
    if args.total_samples is not None and args.total_samples > 0:
        total_target = args.total_samples
    elif args.max_total_samples is not None and args.max_total_samples > 0:
        total_target = args.max_total_samples

    per_cluster_counts: List[int]
    if args.feature_samples_per_cluster is not None and args.feature_samples_per_cluster > 0:
        desired_total = args.feature_samples_per_cluster * num_clusters
        if total_target is not None and total_target < desired_total:
            base = total_target // num_clusters
            remainder = total_target % num_clusters
            per_cluster_counts = [
                base + (1 if idx < remainder else 0)
                for idx in range(num_clusters)
            ]
        else:
            per_cluster_counts = [args.feature_samples_per_cluster] * num_clusters
            total_target = desired_total
    else:
        if total_target is None:
            total_target = len(candidate_pool)
        base = total_target // num_clusters
        remainder = total_target % num_clusters
        per_cluster_counts = [
            base + (1 if idx < remainder else 0)
            for idx in range(num_clusters)
        ]

    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for cluster_id in range(num_clusters):
        cluster_samples = clusters.get(cluster_id, [])
        cluster_samples.sort(key=lambda item: item['score'], reverse=True)
        limit = per_cluster_counts[cluster_id] if cluster_id < len(per_cluster_counts) else 0
        if limit <= 0:
            continue

        chosen = cluster_samples[:limit]
        selected.extend(chosen)
        leftovers.extend(cluster_samples[limit:])

        if len(chosen) < limit:
            print(
                f'Cluster {cluster_id}: only {len(chosen)} samples available '
                f'for the requested {limit}. '
            )
        else:
            print(
                f'Cluster {cluster_id}: selected top {limit} of {len(cluster_samples)} samples.'
            )

    if total_target is not None and len(selected) < total_target:
        needed = total_target - len(selected)
        if leftovers:
            leftovers.sort(key=lambda item: item['score'], reverse=True)
            additions = leftovers[:needed]
            if additions:
                selected.extend(additions)
                print(
                    f'Filled remaining quota with {len(additions)} samples from the leftovers.'
                )
        if len(selected) < total_target:
            print(
                f'Warning: feature-diversity sampling gathered {len(selected)} '
                f'samples but the target was {total_target}.'
            )

    if total_target is not None and len(selected) > total_target:
        selected.sort(key=lambda item: item['score'], reverse=True)
        selected = selected[:total_target]
        print(f'Trimmed selection to the target of {total_target} samples.')

    print(f'Total curated samples: {len(selected)}')
    return selected


def run_meta_label_sampling(
    curator: "UnifiedInformativenessCurator",
    dataset: ImageFolderWithPaths,
    args: argparse.Namespace,
    subset_size: Optional[int],
) -> List[Dict[str, Any]]:
    if args.meta_top_n <= 0:
        raise ValueError('--meta_top_n must be positive when using meta-label sampling.')

    summary_accumulator = ScoreSummaryAccumulator()
    scored_samples: List[Dict[str, Any]] = []

    for sample in curator.score_dataset_iter(
        dataset=dataset,
        subset_size=subset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        top_n=args.meta_top_n,
    ):
        if 'meta_label' not in sample:
            raise RuntimeError('Meta-label information was not produced during scoring.')
        summary_accumulator.update(sample)
        scored_samples.append(dict(sample))

    if not scored_samples:
        raise ValueError('No samples were scored when performing meta-label sampling.')

    summary = summary_accumulator.to_dict()
    print('Scoring summary:')
    for key, value in summary.items():
        print(f'  {key}: {value:.6f}')

    counter: Counter = Counter(sample['meta_label'] for sample in scored_samples)
    most_common = counter.most_common(args.meta_label_top_k)

    if not most_common:
        raise ValueError('No meta-label groups were identified from the teacher predictions.')

    valid_meta_labels = [entry[0] for entry in most_common]
    valid_set = set(valid_meta_labels)

    print('Top meta-label groups:')
    for label, freq in most_common:
        print(f'  {label}: {freq} samples')

    grouped: Dict[Optional[Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for sample in scored_samples:
        meta_label = sample['meta_label']
        group_key: Optional[Tuple[int, ...]]
        if meta_label in valid_set:
            group_key = meta_label
        else:
            group_key = None
        grouped[group_key].append(sample)

    total_target: Optional[int] = None
    if args.total_samples is not None and args.total_samples > 0:
        total_target = args.total_samples
    elif args.max_total_samples is not None and args.max_total_samples > 0:
        total_target = args.max_total_samples

    per_group_counts: List[int]
    if args.meta_samples_per_group is not None and args.meta_samples_per_group > 0:
        desired_total = args.meta_samples_per_group * len(valid_meta_labels)
        if total_target is not None and total_target < desired_total:
            base = total_target // len(valid_meta_labels)
            remainder = total_target % len(valid_meta_labels)
            per_group_counts = [
                base + (1 if idx < remainder else 0)
                for idx in range(len(valid_meta_labels))
            ]
        else:
            per_group_counts = [args.meta_samples_per_group] * len(valid_meta_labels)
            total_target = desired_total
    else:
        if total_target is None:
            raise ValueError(
                'Meta-label sampling requires either --total_samples or --meta_samples_per_group.'
            )
        base = total_target // len(valid_meta_labels)
        remainder = total_target % len(valid_meta_labels)
        per_group_counts = [
            base + (1 if idx < remainder else 0)
            for idx in range(len(valid_meta_labels))
        ]

    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for idx, meta_label in enumerate(valid_meta_labels):
        group_samples = grouped.get(meta_label, [])
        group_samples.sort(key=lambda item: item['score'], reverse=True)
        limit = per_group_counts[idx]
        if limit <= 0:
            continue

        chosen = group_samples[:limit]
        for entry in chosen:
            entry['meta_label'] = list(entry['meta_label'])
        selected.extend(chosen)
        leftovers.extend(group_samples[limit:])

        if len(chosen) < limit:
            print(
                f'Meta-label {meta_label}: only {len(chosen)} samples available '
                f'for the requested {limit}.'
            )
        else:
            print(
                f'Meta-label {meta_label}: selected top {limit} of {len(group_samples)} samples.'
            )

    if total_target is not None and len(selected) < total_target:
        needed = total_target - len(selected)
        supplemental: List[Dict[str, Any]] = []
        if args.meta_include_others:
            others_group = grouped.get(None, [])
            others_group.sort(key=lambda item: item['score'], reverse=True)
            for entry in others_group:
                entry['meta_label'] = list(entry['meta_label']) if isinstance(entry['meta_label'], tuple) else entry['meta_label']
            supplemental.extend(others_group)

        if leftovers:
            leftovers.sort(key=lambda item: item['score'], reverse=True)
            supplemental.extend(leftovers)

        if supplemental:
            additions = supplemental[:needed]
            selected.extend(additions)
            print(f'Added {len(additions)} samples from auxiliary groups to meet the target.')

        if len(selected) < total_target:
            print(
                f'Warning: meta-label sampling gathered {len(selected)} samples '
                f'but the target was {total_target}.'
            )

    if total_target is not None and len(selected) > total_target:
        selected.sort(key=lambda item: item['score'], reverse=True)
        selected = selected[:total_target]
        print(f'Trimmed selection to the target of {total_target} samples.')

    if not selected:
        raise ValueError('Meta-label sampling did not select any samples.')

    for entry in selected:
        if isinstance(entry.get('meta_label'), tuple):
            entry['meta_label'] = list(entry['meta_label'])

    print(f'Total curated samples: {len(selected)}')
    return selected



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

    if args.dataset is not None and args.dataset in CLASSIFICATION_DATASETS:
        num_teacher_classes = CLASSIFICATION_DATASETS[args.dataset]
    else:
        num_teacher_classes = getattr(curator.teacher_model, 'num_classes', None)
        if num_teacher_classes is None:
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
                logits = curator.teacher_model(dummy)
                num_teacher_classes = logits.shape[1]

    if num_teacher_classes is None:
        raise ValueError('Unable to determine the number of pseudo-label classes for distribution.')

    if args.sampling_strategy == 'pseudo_label':
        curated_samples = run_pseudo_label_sampling(
            curator=curator,
            dataset=dataset,
            args=args,
            subset_size=subset_size,
            num_teacher_classes=num_teacher_classes,
        )
    elif args.sampling_strategy == 'feature_diversity':
        curated_samples = run_feature_diversity_sampling(
            curator=curator,
            dataset=dataset,
            args=args,
            subset_size=subset_size,
        )
    elif args.sampling_strategy == 'meta_label':
        curated_samples = run_meta_label_sampling(
            curator=curator,
            dataset=dataset,
            args=args,
            subset_size=subset_size,
        )
    else:
        raise ValueError(f'Unknown sampling strategy: {args.sampling_strategy}')

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

