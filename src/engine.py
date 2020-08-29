import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F
import config
from utils import AverageMeter


def loss_fn(outputs, targets):
    if len(targets.shape) == 1:
        return F.cross_entropy(outputs, targets)
    else:
        return torch.mean(torch.sum(-targets * F.log_softmax(outputs, dim=1), dim=1))


def label_smoothing_loss_fn(outputs, targets, epsilon=0.1):
    num_classes = outputs.shape[1]
    device = outputs.device
    onehot = F.one_hot(targets, num_classes).to(dtype=torch.float, device=device)
    targets = (1 - epsilon) * onehot + torch.ones(onehot.shape).to(device) * epsilon / num_classes
    return loss_fn(outputs, targets)


def train(data_loader, model, optimizer, device, scaler, scheduler=None, clip_grad=False, label_smooth=False,):
    "Runs an epoch of model training"
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    for data in data_loader:

        optimizer.zero_grad()

        # Get data
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            if label_smooth:
                loss = label_smoothing_loss_fn(outputs, targets)
            else:
                loss = loss_fn(outputs, targets)

        # Record training loss and accuracy
        losses.update(loss.item(), len(inputs))
        probs = outputs.detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        targs = targets.detach().cpu().numpy()
        accuracy = metrics.accuracy_score(targs, preds)
        accuracies.update(accuracy, len(inputs))

        # Backward pass
        scaler.scale(loss).backward()
        if clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    return losses.avg, accuracies.avg


def evaluate(data_loader, model, device, target=True):
    "Run evaluation loop"
    final_outputs = []
    final_targets = []
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        for data in data_loader:

            # Get image batch
            if target:
                inputs, targets = data
                targets = targets.to(device)
            else:
                inputs = data
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            if target:
                loss = loss_fn(outputs, targets)
                losses.update(loss.item(), len(outputs))
                targets = targets.detach().cpu().numpy().tolist()
                final_targets.extend(targets)
            outputs = outputs.detach().cpu().numpy().tolist()
            final_outputs.extend(outputs)

    # Return outputs (and targets) as numpy arrays
    final_outputs = np.array(final_outputs)
    if target:
        final_targets = np.array(final_targets)
        return final_outputs, final_targets, losses.avg
    else:
        return final_outputs
