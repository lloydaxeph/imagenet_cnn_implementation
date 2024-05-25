import os
import json

import torch
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
"""
This custom dataset is specifically created for the Mini-ImageNet dataset in Kaggle.
Dataset Download link: https://www.kaggle.com/datasets/deeptrial/miniimagenet/data
"""


class MiniImagenetDataset(Dataset):
    def __init__(self, data_path: str, transform: transforms = None):
        self.data_path = data_path
        self.image_paths = []
        self.labels = []
        self.class_mapping = {}

        class_index_path = os.path.join(data_path, 'imagenet_class_index.json')
        with open(class_index_path, 'r') as f:
            class_id_to_name = json.load(f)
        class_id_to_name = {v[0]: [k, v[1]]for k, v in class_id_to_name.items()}

        image_dir = os.path.join(data_path, 'images')
        for class_name in sorted(os.listdir(image_dir)):
            class_path = os.path.join(image_dir, class_name)
            for image_name in sorted(os.listdir(class_path)):
                image_path = os.path.join(class_path, image_name)
                self.image_paths.append(image_path)

                class_map = class_id_to_name[class_name]
                self.class_mapping[int(class_map[0])] = class_map[1]
                self.labels.append(int(class_map[0]))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> (torch.Tensor, str):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx])

        # Load image as PIL format
        image = Image.open(image_path).convert('RGB')  # Ensure RGB mode

        if self.transform:
            image = self.transform(image)
        return image, label
