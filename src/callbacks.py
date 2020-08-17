import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience=10, mode="max", epsilon=0.0001):
        self.patience = patience
        self.mode = mode
        self.epsilon = epsilon
    
        # Scores
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        if self.mode == "max":
            self.val_score = -np.inf
        else:
            self.val_score = np.inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "max":
            score = epoch_score
        else:
            score = -1. * epoch_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.epsilon:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(epoch_score, model, model_path)

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in (np.inf, -np.inf, np.nan, -np.nan):
            print(f"Validation score improved ({self.val_score} -> {epoch_score}). Saving model...")
            torch.save(model.state_dict, model_path)
        self.val_score = epoch_score
