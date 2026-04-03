import numpy as np
import torch


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """
    Train for one epoch and return average loss.
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    """
    Evaluate on a dataset and return:
        avg_loss
        pred_raw: predictions transformed back to original scale
        y_true: original target values
    """
    model.eval()

    total_loss = 0.0
    total_count = 0

    pred_raw_all = []
    y_true_all = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred_log = model(x)
        loss = loss_fn(pred_log, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        # Convert back from log-space to the original scale.
        pred_raw = torch.expm1(pred_log).cpu().numpy()
        y_true = y.cpu().numpy()

        pred_raw_all.append(pred_raw)
        y_true_all.append(y_true)

    avg_loss = total_loss / max(total_count, 1)
    pred_raw_all = np.concatenate(pred_raw_all)
    y_true_all = np.concatenate(y_true_all)

    return avg_loss, pred_raw_all, y_true_all