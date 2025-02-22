import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input, target, mask):
        loss = self.criterion(input, target)
        loss = loss * mask

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.sum(loss) / torch.sum(mask)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
