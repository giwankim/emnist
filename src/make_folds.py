import pandas as pd
from sklearn import model_selection
import config
import utils


if __name__ == "__main__":
    # For reproducibility
    utils.seed_everything(config.SEED)

    # Get data dataframe
    df = pd.read_csv(config.TRAIN_CSV)

    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # KFold
    df["kfold"] = -1
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=df.digit)):
        df.loc[valid_idx, "kfold"] = fold

    # Save kfolds CSV
    df.to_csv(config.DATA_PATH / "train_folds.csv", index=False)
