import numpy as np
import torch


def FoM(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    prev: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
):
    if mask is not None:
        pred = pred * mask
        gt = gt * mask
        prev = prev * mask

    hit = (gt != prev) & (pred != prev)
    miss = (gt != prev) & (pred == prev)
    false_alarm = (gt == prev) & (pred != prev)

    hit = int(hit.sum())
    miss = int(miss.sum())
    false_alarm = int(false_alarm.sum())

    fom = hit / (hit + miss + false_alarm) if hit + miss + false_alarm != 0 else None

    return fom, (hit, miss, false_alarm)


def FoM_Prob(
    prob: torch.Tensor,
    gt: torch.Tensor,
    prev: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float = 0.5,
):
    pred = torch.where(
        torch.sigmoid(prob) < threshold, torch.tensor(0.0), torch.tensor(1.0)
    )
    return FoM(pred, gt, prev, mask)


def FoM_Float(
    pred: torch.Tensor, gt: torch.Tensor, prev: torch.Tensor, mask: torch.Tensor
):
    ones_tensor = torch.ones_like(pred)

    pred = pred * mask
    gt = gt * mask
    prev = prev * mask

    gt_changed = gt - prev
    gt_unchanged = ones_tensor - gt_changed
    pred_changed = pred - prev
    pred_unchanged = ones_tensor - pred_changed

    hit = gt_changed * pred_changed
    miss = gt_changed * pred_unchanged
    false_alarm = gt_unchanged * pred_changed

    fom = torch.sum(hit) / (
        torch.sum(hit) + torch.sum(miss) + torch.sum(false_alarm) + 1e-8
    )

    return fom
