import numpy as np

import albumentations

import torch

import config
import dataset


class EMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, df, augs=None):
        self.images = df[config.PIXEL_COLS].values
        self.digits = df.digit.values
        self.letters = df.letter.map(config.LETTER_TO_INDEX).values

        if augs is None:
            self.augs = albumentations.Compose([
                albumentations.Normalize(config.MEAN, config.STD, max_pixel_value=255.0, always_apply=True),
            ])
        else:
            self.augs = augs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]

        # image = np.where(image >= 150, image, 0)

        image = image.astype(np.uint8)
        image = image.reshape(28, 28, 1)

        augmented = self.augs(image=image)
        image = augmented["image"]

        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)

        digit = self.digits[item]
        digit = torch.tensor(digit, dtype=torch.long)

        letter = self.letters[item]
        letter = torch.tensor(letter, dtype=torch.long)

        return {
            "image": image, "digit": digit, "letter": letter,
        }


class EMNISTTestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.images = df[config.PIXEL_COLS].values
        self.augs = albumentations.Compose([
            albumentations.Normalize(config.MEAN, config.STD, max_pixel_value=255.0, always_apply=True),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        image = image.reshape(28, 28, 1)
        image = image.astype(np.uint8)

        augmented = self.augs(image=image)
        image = augmented["image"]

        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)

        return {"image": image}
