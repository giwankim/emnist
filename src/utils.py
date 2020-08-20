import random
import numpy as np
import tensorflow as tf
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def onehot(target, num_classes):
    ohe_target = torch.zeros(num_classes, dtype=torch.float)
    ohe_target[target] = 1.0
    return ohe_target
