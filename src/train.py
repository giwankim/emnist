import pandas as pd

import torch

import config
import dataset
import engine
import model


def run():
    pass


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_CSV)
    df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, stratify=df.digit)

    augs = albumentations.Compose(
        [
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.9,
            ),
            albumentations.OneOf([RandomGamma(), RandomBrightnessContrast()]),
            albumentations.Normalize(config.MEAN, config.STD, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_dataset = dataset.EMNISTDataset(df, augs=augs)



    device = torch.device(config.DEVICE)

    # Get model
    model = models.Model()
    model.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters())

    # Run epochs
    for epoch in range(config.EPOCHS):
        pass
