import string


# Data
DATA_PATH = "../input/"
TRAIN_CSV = DATA_PATH + "train.csv"
TEST_CSV = DATA_PATH + "test.csv"
OUTPUT_PATH = "../output/"
SIZE = 28
PIXEL_COLS = [str(i) for i in range(SIZE * SIZE)]

LETTER_TO_INDEX = {c: i for i, c in enumerate(string.ascii_uppercase)}

MEAN = 0.143
STD = 0.253

# Train
DEVICE = "cuda"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
EPOCHS = 150
PATIENCE = 10
