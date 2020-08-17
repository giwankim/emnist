import string


# Data paths
DATA_PATH = "../input/"
TRAIN_CSV = DATA_PATH + "train.csv"
TEST_CSV = DATA_PATH + "test.csv"
OUTPUT_PATH = "../output/"

# Image stats
SIZE = 28
MEAN = 0.143
STD = 0.254

# Dataframe
PIXEL_COLS = [str(i) for i in range(SIZE * SIZE)]
LETTER_TO_INDEX = {c: i for i, c in enumerate(string.ascii_uppercase)}

# Train
DEVICE = "cuda"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
EPOCHS = 150
PATIENCE = 10
CLIP_GRAD = 1.0
