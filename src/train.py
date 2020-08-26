import argparse
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations as A
from sklearn import metrics
from sklearn import model_selection

import torch
import torchcontrib
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

import callbacks
import config
import dataset
import engine
import models
import utils
import model_dispatcher


def run(df, fold, train_idx, valid_idx, model, device):
    # DATASETS
    train_dataset = dataset.EMNISTDataset(df, train_idx)
    valid_dataset = dataset.EMNISTDataset(df, valid_idx)

    # Get dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.9e-3)

    # STOCHASTIC WEIGHT AVERAGING
    # optimizer = torchcontrib.optim.SWA(optimizer, swa_start=20, swa_freq=5, swa_lr=1e-3)
    swa_start = 20
    swa_scheduler = SWALR(
        optimizer, anneal_strategy="cos", anneal_epochs=swa_start, swa_lr=9e-5
    )
    swa_model = AveragedModel(model)
    swa_model.to(device)

    # LEARNING RATE SCHEDULER
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="max", verbose=True, patience=config.LR_REDUCE_PATIENCE, factor=0.5,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS,
    )

    # Early Stopping
    early_stop = callbacks.EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE, mode="max"
    )

    # Gradient scaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    # Model path
    model_path = (
        config.MODEL_PATH / f"{args.model}-fold{fold}-{config.MAJOR}-{config.MINOR}.pt"
    )

    # Run epochs
    for epoch in range(config.EPOCHS):
        # TRAIN FOR LOOP
        engine.train(train_loader, model, optimizer, device, scaler=scaler)

        # VALIDATION
        preds, targs = engine.evaluate(valid_loader, model, device)
        preds = np.argmax(preds, axis=1)
        val_accuracy = metrics.accuracy_score(targs, preds)

        # scheduler.step()
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # scheduler.step(val_accuracy)
            scheduler.step()

        # early stopping if necessary
        early_stop(epoch, val_accuracy, model, model_path)
        if early_stop.early_stop:
            print(
                f"Early stopping. Best score={early_stop.best_score} at epoch {early_stop.best_epoch}. Loading weights..."
            )
            model.load_state_dict(torch.load(model_path))
            break

    # UPDATE BATCH NORMALIZATION STATISTICS
    swa_model = swa_model.cpu()
    update_bn(train_loader, swa_model)
    swa_model.to(device)

    torch.save(swa_model.state_dict(), model_path)

    valid_preds, valid_targs = engine.evaluate(valid_loader, swa_model, device)
    valid_accuracy = metrics.accuracy_score(valid_targs, np.argmax(valid_preds, axis=1))

    return valid_preds


if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser(description="Arguments for training EMNIST model")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Seed randomness")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--model", type=str, default="baseline", help="Name of the model to use"
    )
    args = parser.parse_args()

    # SETUP
    utils.seed_everything(args.seed)
    device = torch.device(config.DEVICE)

    # DATA
    df = pd.read_csv(config.TRAIN_CSV)

    # OOF
    oof = np.zeros((len(df), config.NUM_CLASSES))

    # KFold
    kf = model_selection.StratifiedKFold(n_splits=config.NUM_FOLDS)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=df.digit)):

        # Get model
        model = model_dispatcher.get_model(args.model)
        model.to(device)

        # OOF prediction
        oof[valid_idx] = run(df, fold, train_idx, valid_idx, model, device)

        # CLEANUP
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Save OOF to disk
    np.save(config.OUTPUT_PATH / f"oof-{args.model}-{config.MAJOR}-{config.MINOR}", oof)

    # CV score
    oofb = np.argmax(oof, axis=1)
    accuracy = metrics.accuracy_score(df.digit.values, oofb)
    print(f"CV accuracy score={accuracy:.5f}")

    print(metrics.confusion_matrix(df.digit.values, oofb))
