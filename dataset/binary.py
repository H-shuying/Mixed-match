"""Dataset helpers for binary MixMatch experiments.

This module provides utility functions to load common vision datasets and
convert them into binary classification problems (even labels are treated as
positives, odd labels as negatives).  It also supports sampling datasets with a
user-specified positive class prior in order to study model robustness under
class imbalance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass
class SplitStatistics:
    """Container describing the class balance of a dataset split."""

    total: int
    positive: int
    negative: int

    @property
    def prior(self) -> float:
        if self.total == 0:
            return 0.0
        return self.positive / float(self.total)


class TransformTwice:
    """Apply the same transform function twice.

    MixMatch requires two augmented views of the same input for the unlabeled
    batch.  This helper mirrors the original implementation's behaviour.
    """

    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, inp):
        return self.transform(inp), self.transform(inp)


class _BaseBinaryDataset(Dataset):
    """Base wrapper that applies transforms and binary target mapping."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        image_transform: Callable | None,
        target_transform: Callable[[int], int],
    ):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def _load_example(self, index: int) -> Tuple[Image.Image, int]:
        image, target = self.base_dataset[index]
        image = _ensure_pil_image(image)
        target = _ensure_int(target)
        return image, target


class BinaryLabeledDataset(_BaseBinaryDataset):
    """Dataset wrapper for labeled samples."""

    def __getitem__(self, item: int):  # type: ignore[override]
        index = int(self.indices[item])
        image, target = self._load_example(index)
        if self.image_transform is not None:
            image = self.image_transform(image)
        target = self.target_transform(target)
        return image, target


class BinaryUnlabeledDataset(_BaseBinaryDataset):
    """Dataset wrapper for unlabeled samples returning two views."""

    def __getitem__(self, item: int):  # type: ignore[override]
        index = int(self.indices[item])
        image, target = self._load_example(index)
        transformed = (
            self.image_transform(image)
            if self.image_transform is not None
            else image
        )
        return transformed, self.target_transform(target)


class BinaryEvalDataset(_BaseBinaryDataset):
    """Dataset wrapper used for validation and test splits."""

    def __getitem__(self, item: int):  # type: ignore[override]
        index = int(self.indices[item])
        image, target = self._load_example(index)
        if self.image_transform is not None:
            image = self.image_transform(image)
        target = self.target_transform(target)
        return image, target


def get_binary_datasets(
    name: str,
    root: str,
    n_labeled: int,
    pos_prior: float,
    seed: int = 0,
    val_ratio: float = 0.1,
) -> Tuple[
    BinaryLabeledDataset,
    BinaryUnlabeledDataset,
    BinaryEvalDataset,
    BinaryEvalDataset,
    int,
    Dict[str, SplitStatistics],
]:
    """Create binary datasets for MixMatch training.

    Args:
        name: Dataset identifier. Supported values are ``mnist``,
            ``fashionmnist``, ``svhn``, ``cifar10``, ``cifar100`` and ``stl10``.
        root: Directory where datasets are stored.
        n_labeled: Number of labeled examples available for training.
        pos_prior: Desired prior probability for the positive class. The
            function will attempt to honour this prior for the labeled,
            unlabeled, validation and test splits by sampling from the available
            data.
        seed: Random seed used for shuffling indices.
        val_ratio: Fraction of the training set reserved for validation.

    Returns:
        A tuple containing the labeled training dataset, unlabeled training
        dataset, validation dataset, test dataset, the number of classes (which
        is always two for the binary setting) and statistics describing the
        class distribution of each split.
    """

    name = name.lower()
    if pos_prior < 0.0 or pos_prior > 1.0:
        raise ValueError("pos_prior must be in the range [0, 1]")

    base_train, base_test = _load_base_datasets(name, root)
    train_targets = _binary_targets(base_train, name)
    test_targets = _binary_targets(base_test, name)

    rng = np.random.RandomState(seed)

    labeled_indices, unlabeled_indices, val_indices = _split_train_indices(
        train_targets,
        n_labeled,
        pos_prior,
        val_ratio,
        rng,
    )
    test_indices = _sample_indices_with_prior(
        np.where(test_targets == 1)[0],
        np.where(test_targets == 0)[0],
        len(test_targets),
        pos_prior,
        rng,
    )

    train_transform, eval_transform = _build_transforms(name)

    labeled_dataset = BinaryLabeledDataset(
        base_train,
        labeled_indices,
        train_transform,
        lambda t: _binary_label(name, t),
    )
    unlabeled_dataset = BinaryUnlabeledDataset(
        base_train,
        unlabeled_indices,
        TransformTwice(train_transform),
        lambda _: -1,
    )
    val_dataset = BinaryEvalDataset(
        base_train,
        val_indices,
        eval_transform,
        lambda t: _binary_label(name, t),
    )
    test_dataset = BinaryEvalDataset(
        base_test,
        test_indices,
        eval_transform,
        lambda t: _binary_label(name, t),
    )

    stats = {
        "labeled": _compute_stats(labeled_indices, train_targets),
        "unlabeled": _compute_stats(unlabeled_indices, train_targets),
        "val": _compute_stats(val_indices, train_targets),
        "test": _compute_stats(test_indices, test_targets),
    }

    return (
        labeled_dataset,
        unlabeled_dataset,
        val_dataset,
        test_dataset,
        2,
        stats,
    )


def _compute_stats(indices: Sequence[int], targets: np.ndarray) -> SplitStatistics:
    if len(indices) == 0:
        return SplitStatistics(total=0, positive=0, negative=0)
    selected = targets[np.asarray(indices, dtype=np.int64)]
    pos = int(selected.sum())
    total = int(len(indices))
    neg = total - pos
    return SplitStatistics(total=total, positive=pos, negative=neg)


def _split_train_indices(
    targets: np.ndarray,
    n_labeled: int,
    pos_prior: float,
    val_ratio: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_indices = np.where(targets == 1)[0]
    neg_indices = np.where(targets == 0)[0]
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    pos_ptr = 0
    neg_ptr = 0

    def take(count: int) -> np.ndarray:
        nonlocal pos_ptr, neg_ptr
        if count <= 0:
            return np.empty((0,), dtype=np.int64)

        max_pos = len(pos_indices) - pos_ptr
        max_neg = len(neg_indices) - neg_ptr

        n_pos = int(round(count * pos_prior))
        n_neg = count - n_pos

        n_pos = min(max_pos, n_pos)
        n_neg = min(max_neg, n_neg)

        taken: List[int] = []
        if n_pos > 0:
            taken.extend(pos_indices[pos_ptr : pos_ptr + n_pos])
        if n_neg > 0:
            taken.extend(neg_indices[neg_ptr : neg_ptr + n_neg])

        pos_ptr += n_pos
        neg_ptr += n_neg

        # Fill the remainder if we were unable to satisfy the prior exactly.
        while len(taken) < count and pos_ptr < len(pos_indices):
            taken.append(int(pos_indices[pos_ptr]))
            pos_ptr += 1
        while len(taken) < count and neg_ptr < len(neg_indices):
            taken.append(int(neg_indices[neg_ptr]))
            neg_ptr += 1

        if len(taken) < count:
            # Out of data for one of the classes; return as many samples as
            # possible. This keeps behaviour well-defined even for extreme priors.
            pass

        result = np.asarray(taken, dtype=np.int64)
        rng.shuffle(result)
        return result

    val_count = int(round(len(targets) * val_ratio))
    labeled_indices = take(n_labeled)
    val_indices = take(val_count)

    # All remaining indices become unlabeled samples.
    remaining_pos = pos_indices[pos_ptr:]
    remaining_neg = neg_indices[neg_ptr:]
    unlabeled_count = len(remaining_pos) + len(remaining_neg)
    if unlabeled_count > 0:
        unlabeled_indices = np.concatenate([remaining_pos, remaining_neg])
        rng.shuffle(unlabeled_indices)
    else:
        unlabeled_indices = np.empty((0,), dtype=np.int64)

    return labeled_indices, unlabeled_indices, val_indices


def _sample_indices_with_prior(
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
    count: int,
    pos_prior: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)

    pos_indices = np.asarray(pos_indices, dtype=np.int64)
    neg_indices = np.asarray(neg_indices, dtype=np.int64)
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    max_pos = len(pos_indices)
    max_neg = len(neg_indices)
    target_pos = int(round(count * pos_prior))
    target_neg = count - target_pos

    n_pos = min(max_pos, target_pos)
    n_neg = min(max_neg, target_neg)

    selected: List[int] = []
    if n_pos > 0:
        selected.extend(pos_indices[:n_pos])
    if n_neg > 0:
        selected.extend(neg_indices[:n_neg])

    pos_ptr = n_pos
    neg_ptr = n_neg

    while len(selected) < count and pos_ptr < max_pos:
        selected.append(int(pos_indices[pos_ptr]))
        pos_ptr += 1
    while len(selected) < count and neg_ptr < max_neg:
        selected.append(int(neg_indices[neg_ptr]))
        neg_ptr += 1

    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def _load_base_datasets(name: str, root: str) -> Tuple[Dataset, Dataset]:
    if name == "mnist":
        train = datasets.MNIST(root, train=True, download=True, transform=None)
        test = datasets.MNIST(root, train=False, download=True, transform=None)
    elif name == "fashionmnist":
        train = datasets.FashionMNIST(root, train=True, download=True, transform=None)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=None)
    elif name == "svhn":
        train = datasets.SVHN(root, split="train", download=True, transform=None)
        test = datasets.SVHN(root, split="test", download=True, transform=None)
    elif name == "cifar10":
        train = datasets.CIFAR10(root, train=True, download=True, transform=None)
        test = datasets.CIFAR10(root, train=False, download=True, transform=None)
    elif name == "cifar100":
        train = datasets.CIFAR100(root, train=True, download=True, transform=None)
        test = datasets.CIFAR100(root, train=False, download=True, transform=None)
    elif name == "stl10":
        train = datasets.STL10(root, split="train", download=True, transform=None)
        test = datasets.STL10(root, split="test", download=True, transform=None)
    else:
        raise ValueError(
            "Unsupported dataset '{}'. Available options are: mnist, "
            "fashionmnist, svhn, cifar10, cifar100 and stl10.".format(name)
        )
    return train, test


def _build_transforms(name: str) -> Tuple[Callable, Callable]:
    mean, std = _dataset_stats(name)

    if name in {"mnist", "fashionmnist"}:
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif name in {"svhn", "cifar10", "cifar100"}:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif name == "stl10":
        train_transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError(f"Unsupported dataset '{name}'")

    return train_transform, eval_transform


def _dataset_stats(name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if name in {"mnist", "fashionmnist"}:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif name == "svhn":
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    elif name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif name == "stl10":
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2241, 0.2215, 0.2239)
    else:
        raise ValueError(f"Unsupported dataset '{name}'")
    return mean, std


def _binary_targets(dataset: Dataset, name: str) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError("Dataset does not expose targets or labels attribute")

    if torch.is_tensor(targets):
        targets = targets.numpy()
    targets = np.asarray(targets)

    vectorized = np.vectorize(lambda t: _binary_label(name, t))
    return vectorized(targets).astype(np.int64)


def _binary_label(name: str, target: int) -> int:
    value = _ensure_int(target)
    if name == "svhn" and value == 10:
        value = 0
    return 1 if value % 2 == 0 else 0


def _ensure_pil_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if torch.is_tensor(image):
        image = image.numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image.astype(np.uint8), mode="L")
        if image.ndim == 3 and image.shape[0] in {1, 3} and image.dtype != np.uint8:
            # Tensor-like in CHW format.
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
            return Image.fromarray(image.astype(np.uint8), mode="L")
        return Image.fromarray(image.astype(np.uint8))
    raise TypeError(f"Unsupported image type: {type(image)}")


def _ensure_int(value) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if torch.is_tensor(value):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.squeeze().item())
    raise TypeError(f"Cannot convert {type(value)} to int")
