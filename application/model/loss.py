import torch
import pytorch_lightning as pl


class DiceLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.smooth = 0.5

    def forward(self, pred, target):
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))
