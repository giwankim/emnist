import argparse

import numpy as np
import pandas as pd

import torch

import config
import dataset
import engine
import model_dispatcher


if __name__ == "__main__":
    # Get cli arguments
    parser = argparse.ArgumentParser(description="Arguments for test data inference")
    parser.add_argument("--model", type=str, default="baseline", help="Model name")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    df_test = pd.read_csv(config.TEST_CSV)

    preds = np.zeros((len(df_test), config.NUM_CLASSES))
    for fold in range(config.NUM_FOLDS):
        model = model_dispatcher.get_model(args.model)
        model_path = (
            config.MODEL_PATH
            / f"{args.model}-fold{fold}-{config.MAJOR}-{config.MINOR}.pt"
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        test_dataset = dataset.EMNISTDataset(
            df_test, np.arange(len(df_test)), label=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.TEST_BATCH_SIZE
        )
        preds += engine.evaluate(test_loader, model, device, target=False)

    preds = np.argmax(preds, axis=1)
    submission = pd.DataFrame({"id": df_test.id, "digit": preds})
    submission.to_csv(
        config.OUTPUT_PATH / f"{args.model}-{config.MAJOR}-{config.MINOR}.csv",
        index=False,
    )
    print(submission.head())
