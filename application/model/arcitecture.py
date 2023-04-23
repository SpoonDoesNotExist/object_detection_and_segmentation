import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.loss import DiceLoss


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()

        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, out_channels, upsample=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, out_channels)
        self.down1 = Down(out_channels, out_channels * 2)
        self.down2 = Down(out_channels * 2, out_channels * 4)
        self.down3 = Down(out_channels * 4, out_channels * 8)
        self.down4 = Down(out_channels * 8, out_channels * 8)
        self.up1 = Up(out_channels * 8 * 2, out_channels * 4, upsample)
        self.up2 = Up(out_channels * 4 * 2, out_channels * 2, upsample)
        self.up3 = Up(out_channels * 2 * 2, out_channels, upsample)
        self.up4 = Up(out_channels * 2, out_channels, upsample)
        self.outc = nn.Conv2d(out_channels, n_classes, kernel_size=1)

        self.preprocess = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.loss_fn = DiceLoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def training_step(self, batch):
        x, y = batch['image'].type(torch.float32), batch['mask'].type(torch.float32)
        x = self.preprocess(x)
        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)

        iou = self._iou_metric(y_hat, y)
        self.log('train_iou', iou, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'].type(torch.float32), batch['mask'].type(torch.float32)
        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)

        iou = self._iou_metric(y_hat, y)
        self.log('val_iou', iou, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _iou_metric(self, y_pred, y_true, smooth=1):
        y_pred = y_pred > 0.5
        y_true = y_true > 0.5
        intersection = (y_pred & y_true).sum().float()
        union = (y_pred | y_true).sum().float()
        iou = (intersection + smooth) / (union + smooth)
        return iou
