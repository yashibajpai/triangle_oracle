from pathlib import Path  
import numpy as np  
import torch  
from torch.utils.data import DataLoader  # dataloader

from triangle_oracle.models.transformer_oracle import EdgeTransformerOracle  # model
from triangle_oracle.models.losses import weighted_log_mse_loss  # loss
from triangle_oracle.training.transformer_dataset import EdgeTransformerDataset, collate_edge_transformer  # dataset
from triangle_oracle.utils.metrics import regression_metrics  # metrics
from triangle_oracle.utils.io import save_npz, ensure_dir, save_json  # io utils
from triangle_oracle.utils.seed import set_seed  # reproducibility


def train_one_epoch(model, loader, optimizer, device):  # one epoch
    model.train()  # training mode

    total_loss = 0.0  # track loss
    total_n = 0  # count samples

    for batch in loader:  # iterate batches
        input_ids = batch["input_ids"].to(device)  # move to device
        attention_mask = batch["attention_mask"].to(device)  # mask
        y = batch["y"].to(device)  # targets

        optimizer.zero_grad()  # reset gradients

        pred_log = model(input_ids=input_ids, attention_mask=attention_mask)  # forward pass
        loss = weighted_log_mse_loss(pred_log, y)  # compute loss

        loss.backward()  # backprop
        optimizer.step()  # update weights

        total_loss += loss.item() * len(y)  # accumulate loss
        total_n += len(y)  # count

    return total_loss / max(total_n, 1)  # return avg loss


@torch.no_grad()
def evaluate(model, loader, device):  # evaluation
    model.eval()  # eval mode

    total_loss = 0.0  # track loss
    total_n = 0  # count samples

    all_pred = []  # predictions
    all_true = []  # true values

    for batch in loader:  # iterate batches
        input_ids = batch["input_ids"].to(device)  # inputs
        attention_mask = batch["attention_mask"].to(device)  # mask
        y = batch["y"].to(device)  # targets

        pred_log = model(input_ids=input_ids, attention_mask=attention_mask)  # forward
        loss = weighted_log_mse_loss(pred_log, y)  # loss

        pred_raw = torch.expm1(pred_log).cpu().numpy()  # convert from log
        true_raw = y.cpu().numpy()  # move to cpu

        total_loss += loss.item() * len(y)  # accumulate
        total_n += len(y)  # count

        all_pred.append(pred_raw)  # store predictions
        all_true.append(true_raw)  # store targets

    avg_loss = total_loss / max(total_n, 1)  # avg loss

    y_pred = np.concatenate(all_pred)  # combine predictions
    y_true = np.concatenate(all_true)  # combine targets

    return avg_loss, y_pred, y_true  # return results