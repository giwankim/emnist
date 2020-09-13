import numpy as np
import scipy
from sklearn import metrics

import torch
import torch.nn.functional as F

import config
import dataset
import models
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
    shuffle = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[shuffle]
    y_a, y_b = y, y[shuffle]
    return mixed_x, y_a, y_b, lam


def rand_bbox(W, H, lam, device):
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w = (W * cut_rat).type(torch.long)
    cut_h = (H * cut_rat).type(torch.long)
    # uniform
    cx = torch.randint(W, (1,), device=device)
    cy = torch.randint(H, (1,), device=device)
    x1 = torch.clamp(cx - cut_w // 2, 0, W)
    y1 = torch.clamp(cy - cut_h // 2, 0, H)
    x2 = torch.clamp(cx + cut_w // 2, 0, W)
    y2 = torch.clamp(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0, p=0.5):
    if np.random.random() > p:
        return x, y, torch.zeros_like(y), 1.0
    W, H = x.size(2), x.size(3)
    shuffle = torch.randperm(x.size(0), device=x.device)
    cutmix_x = x[shuffle]

    lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(x.device)
    # lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device)
    x1, y1, x2, y2 = rand_bbox(W, H, lam, x.device)
    cutmix_x[:, :, x1:x2, y1:y2] = x[shuffle, :, x1:x2, y1:y2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / float(W * H)).item()
    y_a, y_b = y, y[shuffle]
    return cutmix_x, y_a, y_b, lam


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
    cutmix=False,
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
        elif cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            criterion = label_smooth_loss_fn if label_smooth else loss_fn
            if cutmix or mixup:
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
        if cutmix or mixup:
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
                inputs = data
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


def get_tta(
    model, test_df, augs, device, batch_size=1024, n=4, beta=0.25, use_max=False
):
    ds = dataset.EMNISTDataset(test_df, np.arange(len(test_df)), label=False)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    logits = evaluate(dl, model, device, test=True)

    aug_ds = dataset.EMNISTDataset(
        test_df, np.arange(len(test_df)), augs=augs, label=False
    )
    aug_dl = torch.utils.data.DataLoader(
        aug_ds, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    aug_logits = [evaluate(aug_dl, model, device, test=True) for i in range(n)]
    aug_logits = np.concatenate(aug_logits, axis=0)
    aug_logits = aug_logits.max(axis=0) if use_max else aug_logits.mean(axis=0)

    if use_max:
        return np.concatenate([logits, aug_logits], axis=0).max(axis=0)
    else:
        return beta * aug_logits + (1 - beta) * logits


def infer(
    model_names,
    checkpoints,
    test_df,
    augs=None,
    batch_size=1024,
    device=config.DEVICE,
    tta=False,
):
    output = np.zeros((len(test_df), 10))

    test_dataset = dataset.EMNISTDataset(
        test_df, np.arange(len(test_df)), augs=augs, label=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    for model_name, checkpoint in zip(model_names, checkpoints):
        model = models.get_model(model_name).to(device)
        model.load_state_dict(torch.load(checkpoint))

        if tta:
            logits = get_tta(model, test_df, augs, device, batch_size)
        else:
            logits = evaluate(test_loader, model, device, test=True)

        probs = scipy.special.softmax(logits, axis=1)
        output += probs / len(model_names)

    return output
