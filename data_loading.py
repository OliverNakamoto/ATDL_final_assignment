
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class LungSegmentationDataset(Dataset):
    def __init__(self, df=None, base_dir=None, resize=None, transform=None, target_transform=None, both_transform=None):
        self.data = df
        self.base_dir = Path(base_dir)
        self.resize = resize
        self.transform = transform
        self.target_transform = target_transform
        self.both_transform = both_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_relative_path = self.data.iloc[idx, 1]
        mask_relative_path = self.data.iloc[idx, 2]

        img_path = self.base_dir / img_relative_path
        mask_path = self.base_dir / mask_relative_path

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = np.stack([image] * 3, axis=-1)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        mask = mask.long()
        return image, mask

class BreastTumorDataset(Dataset):
    def __init__(self, dataset=None, base_dir=None, transform=None, target_transform=None, both_transform=None):
        self.data = dataset
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.both_transform = both_transform

    def __len__(self):
        return len(self.data)

    def combined_transforms(self, image, mask):
        albumentations_transforms = A.Compose([
            A.RandomSizedCrop(min_max_height=(128,256), width=256, height=256, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])

        transform_train_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        ])
        torchvision_transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.1953, 0.1925, 0.1942]),
        ])

        augmented = albumentations_transforms(image=np.array(image), mask=np.array(mask))
        image = augmented["image"]
 
        image = torchvision_transforms(transforms.ToPILImage()(image))

        if mask is not None:
            mask = transform_train_mask(mask)
        return image, mask


    def __getitem__(self, idx):
        class_tumor, index = self.data[idx]
        img_relative_path = f"{class_tumor}/{class_tumor} ({index}).png"
        mask_relative_path = f"{class_tumor}/{class_tumor} ({index})_mask.png"

        img_path = self.base_dir / img_relative_path
        mask_path = self.base_dir / mask_relative_path

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = np.stack([image] * 3, axis=-1)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        counter = 1
        while True:
            additional_mask_path = self.base_dir / f"{class_tumor}/{class_tumor} ({index})_mask_{counter}.png"
            if additional_mask_path.exists():
                additional_mask = cv2.imread(str(additional_mask_path), cv2.IMREAD_GRAYSCALE)
                mask += additional_mask
                counter += 1
            else:
                break

        map_to_class = {
            "normal": 0,
            "benign": 1,
            "malignant": 2,
        }
        mask = np.where(mask == 255, map_to_class[class_tumor], 0)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
#             print('or this')
#            mask = mask.astype(np.uint8)
#            mask = Image.fromarray(mask)
             mask = self.target_transform(mask)
        else:
#            print('nvm')
            image, mask = self.combined_transforms(image, mask)

        return image, mask
