import os
import random
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    #     tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, is_best, checkpoint="checkpoint", filename="checkpoint.pth"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.apth.join(checkpoint, "model_best.pth"))


def find_lr(
    model, train_loader, loss_fn, init_val=1e-8, final_val=10.0, beta=0.98, plot=True
):
    # Save initial state of the model
    init_state = os.path.join(tempfile.mkdtemp(), "init_state.pt")
    torch.save(model.state_dict(), init_state)

    num_steps = len(train_loader)
    mult = (final_val / init_val) ** (1 / (num_steps - 1))
    lr = init_val
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.param_groups[0]["lr"] = lr

    ewa_loss = 0.0
    best_loss = np.inf
    log_lrs = []
    losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # Forwardprop
        inp, targ = batch
        inp = inp.to(device)
        targ = targ.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = loss_fn(out, targ)

        # Update smoothed and debiased losses
        ewa_loss = beta * ewa_loss + (1 - beta) * loss.item()
        debiased_loss = ewa_loss / (1 - beta ** (batch_idx + 1))

        # Stop if loss is exploding
        if batch_idx and debiased_loss > 4 * best_loss:
            break
        # Update `best_loss`
        best_loss = min(best_loss, debiased_loss)

        # Record values
        losses.append(debiased_loss)
        log_lrs.append(np.log10(lr))

        # Backprop
        loss.backward()
        optimizer.step()

        # Update LR
        lr *= mult
        optimizer.param_groups[0]["lr"] = lr

    # Reset the model to initial state
    model.load_state_dict(torch.load(init_state))

    # Plot
    if plot:
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.show()

    return log_lrs, losses
