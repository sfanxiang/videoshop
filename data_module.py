import copy
import lightning.pytorch as pl
import multiprocess as mp
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PreprocessedDataset(Dataset):
    def __init__(self, dataset, height, width, image_height, image_width, resize=True):
        super().__init__()
        self.dataset = dataset

        self._image1_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_height, image_width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        self._video_transforms = transforms.Compose(
            [
                *(
                    [
                        transforms.Resize(
                            (height, width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                        )
                    ]
                    if resize
                    else []
                ),
                transforms.CenterCrop((height, width)),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self._image2_transforms = transforms.Compose(
            [
                *(
                    [
                        transforms.Resize(
                            (height, width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                        )
                    ]
                    if resize
                    else []
                ),
                transforms.CenterCrop((height, width)),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = copy.copy(self.dataset[idx])

        if "video" in item:
            if item["video"].dtype == torch.uint8:
                item["video"] = item["video"] / 255.0
            item["video"] = self._video_transforms(item["video"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if item["image1"].dtype == torch.uint8:
            item["image1"] = item["image1"] / 255.0
        item["image1"] = self._image1_transforms(item["image1"].permute(2, 0, 1)).permute(1, 2, 0)
        if item["image2"].dtype == torch.uint8:
            item["image2"] = item["image2"] / 255.0
        item["image2"] = self._image2_transforms(item["image2"].permute(2, 0, 1)).permute(1, 2, 0)

        return item
