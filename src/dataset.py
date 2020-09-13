import numpy as np
import albumentations as A

import torch
import torch.nn.functional as F

import config
import dataset
import utils
from config import PIXEL_COLS, SIZE, MEAN, STD, NUM_CLASSES, NUM_FOLDS


class EMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, df, list_IDs, augs=None, label=True, rgb=False):
        self.list_IDs = list_IDs
        self.label = label
        self.rgb = rgb

        self.images = df[PIXEL_COLS].values
        self.images = self.images.astype(np.uint8)
        self.images = self.images.reshape(-1, SIZE, SIZE, 1)

        if label:
            self.digits = df.digit.values

        if augs is None:
            if self.rgb:
                self.augs = A.Compose(
                    [
                        A.Normalize(
                            mean=[MEAN, MEAN, MEAN],
                            std=[STD, STD, STD],
                            max_pixel_value=255.0,
                            always_apply=True,
                        ),
                    ]
                )
            else:
                self.augs = A.Compose(
                    [A.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True,),]
                )
        else:
            self.augs = augs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, item):
        # Get image
        index = self.list_IDs[item]
        image = self.images[index]
        if self.rgb:
            image = np.concatenate([image, image, image], axis=2)

        # Augment image
        image = self.augs(image=image)["image"]

        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)

        # Get labels and return
        if self.label:
            digit = self.digits[index]
            digit = torch.tensor(digit, dtype=torch.long)
            return image, digit
        else:
            return image
