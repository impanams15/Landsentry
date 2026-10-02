"""
Siamese CNN for bitemporal landslide change detection.

Pre- and post-event images are passed through the SAME encoder (shared weights
= the defining property of a Siamese network). Comparing the two resulting
feature stacks, rather than segmenting either image alone, is what lets the
model tell a genuine landslide scar apart from a look-alike (bare farmland,
a quarry, a dry-season vegetation patch) that would fool a single-image model.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SiameseEncoder(nn.Module):
    """Shared-weight encoder applied independently to the pre- and post-event image."""

    def __init__(self, in_channels=4, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base * 4, base * 8)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool1(f1))
        f3 = self.enc3(self.pool2(f2))
        f4 = self.enc4(self.pool3(f3))
        return [f1, f2, f3, f4]  # multi-scale features, shallow -> deep


class SiameseChangeDetector(nn.Module):
    """
    Full Siamese change-detection network.

    1. pre and post pass through the SAME encoder instance (true weight sharing).
    2. At every scale, the absolute difference between pre- and post-event
       features becomes the skip connection into a U-Net-style decoder.
    3. The decoder upsamples back to full resolution and outputs one logit
       per pixel: probability that pixel is newly landslide-affected.
    """

    def __init__(self, in_channels=4, base=32, num_classes=1):
        super().__init__()
        self.encoder = SiameseEncoder(in_channels, base)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 4 * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 2 * 2, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        
        self.dropout = nn.Dropout2d(p=0.2)
        self.classifier = nn.Conv2d(base, num_classes, 1)

    def forward(self, pre, post, return_features=False):
        f_pre = self.encoder(pre)
        f_post = self.encoder(post)  # same weights as f_pre -> the "Siamese" part
        d1, d2, d3, d4 = [torch.abs(a - b) for a, b in zip(f_pre, f_post)]

        x = self.dropout(self.up3(d4))
        x = self.dec3(torch.cat([x, d3], dim=1))
        x = self.dropout(self.up2(x))
        x = self.dec2(torch.cat([x, d2], dim=1))
        x = self.dropout(self.up1(x))
        x = self.dec1(torch.cat([x, d1], dim=1))

        logits = self.classifier(self.dropout(x))  # raw logits, shape [B, 1, H, W]
        
        if return_features:
            return logits, d4
        return logits
