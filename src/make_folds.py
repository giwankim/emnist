import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn import model_selection

import config


def make_folds(df, dir_path):
    df = df.copy()

    dir_path = pathlib.Path(dir_path)

    # Shuffle
    df = df.sample(frac=1.0).reset_index(drop=True)

    # KFold
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (train_indices, valid_indices) in enumerate(
        kf.split(X=df, y=df.digit.values)
    ):
        np.save(dir_path / f"train_idx-train-csv-fold{fold}", train_indices)
        np.save(dir_path / f"valid_idx-train-csv-fold{fold}", valid_indices)


if __name__ == "__main__":
    # CLI ARGUMENTS
    parser = argparse.ArgumentParser(description="Make kfolds of dataframe")
    parser.add_argument("--dataframe", type=str, default=config.TRAIN_CSV)
    parser.add_argument("--path", type=str, default="../input/folds/")
    args = parser.parse_args()

    # Get dataframe
    df = pd.read_csv(args.dataframe)
    # Make folds
    make_folds(df, args.path)
