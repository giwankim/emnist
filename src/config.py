import pathlib

# VERSIONING
MAJOR = 1
MINOR = 0

DEVICE = "cuda"
SEED = 42

# PATHS for data files
DATA_PATH = pathlib.Path("/home/isleof/Development/emnist/input")
OUTPUT_PATH = pathlib.Path("/home/isleof/Development/emnist/output")
MODEL_PATH = pathlib.Path("/home/isleof/Development/emnist/models")
TRAIN_CSV = DATA_PATH / "train.csv"
TEST_CSV = DATA_PATH / "test.csv"

# IMAGE stats
SIZE = 28
MEAN = 0.143
STD = 0.254

# DATAFRAME
PIXEL_COLS = [str(i) for i in range(784)]
LETTER_TO_INDEX = {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
NUM_CLASSES = 10

#### Train/Hyperparameters ####

# Batch Sizes
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128

NUM_FOLDS = 5
EPOCHS = 200

CLIP_GRAD = 1.0
LR_REDUCE_PATIENCE = 10
EARLY_STOP_PATIENCE = 15
