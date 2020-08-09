import torch
import torch.nn.functional as F


def loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets)


def train(data_loader, model, optimizer, device, scheduler=None):
    model.train()

    for data in data_loader:
        inputs = data["image"]
        digits = data["digit"]
        letters = data["letter"]

        inputs = inputs.to(device)
        digits = digits.to(device)
        letters = letters.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, digits)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()


def evaluate(data_loader, model, device):
    model.eval()

    final_outputs = []
    final_targets = []

    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"]
            inputs = inputs.to(device)
            targets = data["digit"]

            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy().tolist()
            targets = targets.numpy().tolist()
            final_outputs.extend(outputs)
            final_targets.extend(targets)

    return final_outputs, final_targets


def infer(data_loader, model, device):
    model.eval()

    final_outputs = []
    for data in data_loader:
        images = data["image"]
        images = images.to(device)

        outputs = model(images)
        outputs = outputs.detach().cpu().numpy().tolist()
        final_outputs.extend(outputs)

    return final_outputs
