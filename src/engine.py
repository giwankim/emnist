import numpy as np
import torch
import torch.nn.functional as F
import config


def loss_fn(outputs, targets):
    if len(targets.shape) == 1:
        return F.cross_entropy(outputs, targets)
    else:
        return torch.mean(torch.sum(-targets * F.log_softmax(outputs, dim=1), dim=1))


def train(data_loader, model, optimizer, device, scaler, clip_grad=False):
    "Runs an epoch of model training"
    model.train()
    for data in data_loader:

        optimizer.zero_grad()

        # Get data batch
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # FORWARD PASS
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # BACKWARD PASS
        # Multiplies loss by scale factor before backward pass
        scaler.scale(loss).backward()

        if clip_grad:
            # Gradient clipping with unscaled gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)

        # Unscale gradients and calls `optimizer.step()`
        scaler.step(optimizer)
        # Updates the scaler's scale factor
        scaler.update()


def evaluate(data_loader, model, device, target=True):
    "Run evaluation loop"
    final_outputs = []
    final_targets = []

    model.eval()

    with torch.no_grad():
        for data in data_loader:

            # Get image batch
            if target:
                inputs, targets = data
            else:
                inputs = data
            inputs = inputs.to(device)

            # Get outputs from model
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy().tolist()
            final_outputs.extend(outputs)

            # Get target for cross-validation metrics
            if target:
                targets = targets.detach().cpu().numpy().tolist()
                final_targets.extend(targets)

    # Return outputs (and targets) as numpy arrays
    final_outputs = np.array(final_outputs)

    if target:
        final_targets = np.array(final_targets)
        return final_outputs, final_targets
    else:
        return final_outputs
