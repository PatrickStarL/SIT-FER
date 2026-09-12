"""RAF-DB dataset loader for semi-supervised training"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .randaugment import RandAugment


class TransformTwice:
    """Wraps a transform to produce (weak, weak, strong) views of the same image"""

    def __init__(self, transform):
        self.transform = transform
        self.strong_transform = transform
        # Prepend RandAugment to a copy of the transform pipeline for the strong view
        import copy
        self.strong_transform = copy.deepcopy(transform)
        self.strong_transform.transforms.insert(0, RandAugment(3, 5))

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        out3 = self.strong_transform(inp)
        return out1, out2, out3


def img_loader(path: str) -> Optional[Image.Image]:
    """Load an image with OpenCV and convert to PIL"""
    try:
        img = cv2.imread(path)
        return Image.fromarray(img)
    except IOError:
        print(f'Cannot load image {path}')
        return None


def target_read(path: str) -> List[int]:
    """Read integer labels from a `<image_path> <label>` list file"""
    with open(path) as f:
        img_label_list = f.read().splitlines()
    return [int(info.split(' ')[1]) for info in img_label_list]


def data_split(filename: str, n_labeled: int, num_classes: int = 7) -> Tuple[List[int], List[int]]:
    """
    Split dataset indices into labeled/unlabeled sets, roughly balanced per class.

    Note: mirrors the original paper's split logic, including the special-cased
    minority class (index 2) which uses a fixed count of 57 labeled samples.
    """
    labels = np.array(target_read(filename))
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    minority_class_count = 57
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if i != 2:
            n_per_class = int((n_labeled - minority_class_count) / (num_classes - 1))
            train_labeled_idxs.extend(idxs[:n_per_class])
            train_unlabeled_idxs.extend(idxs[n_per_class:])
        else:
            train_labeled_idxs.extend(idxs[:minority_class_count])
            train_unlabeled_idxs.extend(idxs[minority_class_count:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    return train_labeled_idxs, train_unlabeled_idxs


class RAFDataset(torch.utils.data.Dataset):
    """Base RAF-DB dataset reading from a `<image_path> <label>` list file"""

    def __init__(self, root: str, file_list: str, transform=None, loader=img_loader):
        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index: int):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index, index

    def __len__(self) -> int:
        return len(self.image_list)


class RAFLabeledDataset(RAFDataset):
    """Labeled subset of RAF-DB, indexed by a fixed index array"""

    def __init__(self, root: str, file_list: str, indexs: Optional[np.ndarray], transform=None):
        super().__init__(root, file_list, transform=transform)

        if indexs is not None:
            self.image_list = np.array(self.image_list)[indexs]
            self.label_list = np.array(self.label_list)[indexs]
            self.indexs = indexs

    def __getitem__(self, index: int):
        img_path = self.image_list[index]
        label = self.label_list[index]
        original_id = self.indexs[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index, original_id


class RAFUnlabeledDataset(RAFLabeledDataset):
    """Unlabeled subset of RAF-DB — labels are masked to -1"""

    def __init__(self, root: str, file_list: str, indexs: Optional[np.ndarray], transform=None):
        super().__init__(root, file_list, indexs, transform=transform)
        self.label_list = np.array([-1 for _ in range(len(self.label_list))])

    def __getitem__(self, index: int):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index, index


def get_raf(
    train_root: str,
    train_file_list: str,
    test_root: str,
    test_file_list: str,
    n_labeled: int,
    transform_train=None,
    transform_val=None,
    num_classes: int = 7,
):
    """
    Build labeled/unlabeled/test datasets for semi-supervised training on RAF-DB.

    Returns:
        (train_labeled_dataset, train_unlabeled_dataset, test_dataset)
    """
    train_labeled_idxs, train_unlabeled_idxs = data_split(
        train_file_list, int(n_labeled), num_classes=num_classes
    )

    train_labeled_dataset = RAFLabeledDataset(
        train_root, train_file_list, train_labeled_idxs, transform=transform_train
    )
    train_unlabeled_dataset = RAFUnlabeledDataset(
        train_root, train_file_list, train_unlabeled_idxs,
        transform=TransformTwice(transform_train)
    )
    test_dataset = RAFDataset(test_root, test_file_list, transform=transform_val)

    print(f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
