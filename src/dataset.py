import numpy as np
import albumentations as A

import torch
import torch.nn.functional as F

import config
import dataset
import utils


class EMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, df, list_IDs, augs=None, label=True, rgb=False):
        self.list_IDs = list_IDs
        self.label = label
        self.rgb = rgb

        self.images = df[config.PIXEL_COLS].values
        self.images = self.images.astype(np.uint8)
        self.images = self.images.reshape(-1, config.SIZE, config.SIZE, 1)

        if label:
            self.digits = df.digit.values

        if augs is None:
            if self.rgb:
                self.augs = A.Compose(
                    [
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value=255.0,
                            always_apply=True,
                        ),
                    ]
                )
            else:
                self.augs = A.Compose(
                    [
                        A.Normalize(
                            config.MEAN,
                            config.STD,
                            max_pixel_value=255.0,
                            always_apply=True,
                        ),
                    ]
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


class Mixup(torch.utils.data.Dataset):
    def __init__(
        self, dataset, num_classes=config.NUM_CLASSES, num_mix=1, beta=1.0, prob=0.5
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, target = self.dataset[item]
        # ohe_target = utils.onehot(target, self.num_classes)
        ohe_target = F.one_hot(target, self.num_classes).type(torch.float)

        for _ in range(self.num_mix):
            r = np.random.random()
            if r > self.prob:
                continue

            # Get mixup parameters
            if self.beta > 0:
                lam = np.random.beta(self.beta, self.beta)
            else:
                lam = 1.0
            rand_index = np.random.choice(len(self))

            image2, target2 = self.dataset[rand_index]
            # ohe_target2 = utils.onehot(target2, self.num_classes)
            ohe_target2 = F.one_hot(target2, self.num_classes).type(torch.float)

            # Mixup
            image = lam * image + (1 - lam) * image2
            ohe_target = lam * ohe_target + (1 - lam) * ohe_target2

        return image, ohe_target
