import math
import warnings

import torch.nn

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnelingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between `warmup_start_lr` and `base_lr` followed by cosine annealing schedule between
    base_lr and eta_min
    """
    def __init__(self, optimizer, warmup_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):

        super(LinearWarmupCosineAnnelingLR, self).__init__(optimizer, last_epoch)