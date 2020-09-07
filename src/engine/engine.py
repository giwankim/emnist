import numpy as np
from sklearn import metrics

import torch
import torch.nn.functional as F

import config
import utils


def loss_fn(outputs, targets):
    if len(targets.shape) == 1:
        return F.cross_entropy(outputs, targets)
    else:
        return torch.mean(torch.sum(-targets * F.log_softmax(outputs, dim=1), dim=1))


def label_smooth_loss_fn(outputs, targets, epsilon=0.1):
    num_classes = outputs.shape[1]
    device = outputs.device
    onehot = F.one_hot(targets, num_classes).to(dtype=torch.float, device=device)
    targets = (1 - epsilon) * onehot + torch.ones(onehot.shape).to(
        device
    ) * epsilon / num_classes
    return loss_fn(outputs, targets)


def mixup_data(x, y, alpha=0.4, p=0.5):
    if np.random.random() > p:
        return x, y, torch.zeros_like(y), 1.0
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    shuffle = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[shuffle]
    y_a, y_b = y, y[shuffle]
    return mixed_x, y_a, y_b, lam


def train(
    data_loader,
    model,
    optimizer,
    device,
    scaler,
    scheduler=None,
    clip_grad=False,
    label_smooth=False,
    mixup=False,
):
    "Runs an epoch of model training"
    correct = 0
    total = 0
    total_loss = 0

    model.train()
    for data in data_loader:

        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # Get data
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            criterion = label_smooth_loss_fn if label_smooth else loss_fn
            if mixup:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )
            else:
                loss = criterion(outputs, targets)

        # Backward pass
        scaler.scale(loss).backward()
        if clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Record metrics
        preds = torch.argmax(outputs, dim=1)
        if mixup:
            correct += (lam * preds.eq(targets_a).cpu().sum().float()) + (
                (1 - lam) * preds.eq(targets_b).cpu().sum().float()
            )
        else:
            correct += preds.eq(targets).cpu().sum().float()
        total += targets.size(0)
        total_loss += loss.item() * len(inputs)

    return total_loss / total, correct / total


def evaluate(data_loader, model, device, test=False):
    "Run evaluation loop"
    final_outputs = []
    final_targets = []
    total_loss = 0
    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for data in data_loader:

            # Get image batch
            if test:
                intputs = data
            else:
                inputs, targets = data
                targets = targets.to(device)
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            final_outputs.append(outputs.detach().cpu().numpy())
            if not test:
                final_targets.append(targets.detach().cpu().numpy())
                loss = loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1)
                total_loss += loss.item() * targets.size(0)
                total += targets.size(0)
                correct += preds.eq(targets).cpu().sum().float()

    final_outputs = np.concatenate(final_outputs, axis=0)
    if test:
        return final_outputs
    else:
        final_targets = np.concatenate(final_targets, axis=0)
        return final_outputs, final_targets, total_loss / total, correct / total

