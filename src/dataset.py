import numpy as np
import albumentations as A
import torch

import config
import dataset


class EMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, df, list_IDs, augs=None, label=True):
        self.list_IDs = list_IDs
        self.label = label
        self.images = df[config.PIXEL_COLS].values
        self.images = self.images.astype(np.uint8)
        self.images = self.images.reshape(-1, config.SIZE, config.SIZE, 1)

        if label:
            self.digits = df.digit.values
            self.letters = df.letter.map(config.LETTER_TO_INDEX).values

        if augs is None:
            self.augs = A.Compose([
                A.Normalize(config.MEAN, config.STD, max_pixel_value=255.0, always_apply=True),
            ])
        else:
            self.augs = augs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, item):
        # Get image
        index = self.list_IDs[item]
        image = self.images[index]

        # Augment image
        image = self.augs(image=image)["image"]

        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)

        # Get labels and return
        if self.label:
            digit = self.digits[index]
            letter = self.letters[index]
            digit = torch.tensor(digit, dtype=torch.long)
            letter = torch.tensor(letter, dtype=torch.long)
            return {"image": image, "digit": digit, "letter": letter,}
        else:
            return {"image": image}


# class EMNISTTestDataset(torch.utils.data.Dataset):
#     def __init__(self, df, augs=None):
#         self.images = df[config.PIXEL_COLS].values
#         if augs is None:
#             self.augs = A.Compose([
#                 A.Normalize(config.MEAN, config.STD, max_pixel_value=255.0, always_apply=True),
#             ])
#         else:
#             self.augs = augs

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, item):
#         image = self.images[item]
#         image = image.reshape(28, 28, 1)
#         image = image.astype(np.uint8)

#         augmented = self.augs(image=image)
#         image = augmented["image"]

#         image = torch.tensor(image, dtype=torch.float)
#         image = image.permute(2, 0, 1)

#         return {"image": image}
