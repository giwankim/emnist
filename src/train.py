import argparse

import torch

import config
import utils

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
