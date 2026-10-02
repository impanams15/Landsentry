"""
Transformer-based bitemporal change detector (BIT/ChangeFormer-style),
built to be trained and benchmarked directly against the Siamese CNN on the
same patches.

Pipeline:
    1. A shared CNN backbone extracts a compact feature map for each of the
       pre- and post-event images (same idea as the Siamese encoder, but
       shallower, since the transformer does the heavy lifting on context).
    2. Each feature map is compressed into a handful of "semantic tokens"
       (learned spatial-attention pooling) -- this is what makes the
       transformer step affordable at image resolution.
    3. A transformer encoder is applied jointly to the pre- and post-event
       tokens, so each token attends over the full bitemporal context
       (this is the key difference from the Siamese CNN, which only ever
       compares features at the same spatial location).
    4. Context-enriched tokens are projected back onto the pixel grid via
       cross-attention (transformer "decoder" in the BIT sense).
    5. The absolute difference between the two enriched feature maps is
       upsampled back to full resolution and classified pixel-wise.
"""
import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """Lightweight shared backbone: two stride-2 conv blocks (4x spatial downsample)."""

    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # [B, out_channels, H/4, W/4]


class SemanticTokenizer(nn.Module):
    """Pools a spatial feature map into `num_tokens` tokens via learned spatial attention."""

    def __init__(self, in_channels, num_tokens=4):
        super().__init__()
        self.token_proj = nn.Conv2d(in_channels, num_tokens, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        attn = self.token_proj(x)                 # [B, L, H, W]
        L = attn.shape[1]
        attn = torch.softmax(attn.view(B, L, H * W), dim=-1)  # spatial softmax per token
        x_flat = x.view(B, C, H * W)
        tokens = torch.einsum("blk,bck->blc", attn, x_flat)   # [B, L, C]
        return tokens


class TransformerContext(nn.Module):
    """
    Standard transformer encoder applied to the concatenated pre+post tokens,
    so every token attends over BOTH time steps -- this is the "bitemporal
    context modeling" step that distinguishes this architecture from the
    Siamese CNN's purely local feature comparison.
    """

    def __init__(self, dim, depth=2, heads=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens_pre, tokens_post):
        L = tokens_pre.shape[1]
        joint = torch.cat([tokens_pre, tokens_post], dim=1)  # [B, 2L, C]
        joint = self.transformer(joint)
        return joint[:, :L, :], joint[:, L:, :]


class TokenDecoder(nn.Module):
    """Projects context-enriched tokens back onto the pixel grid via cross-attention."""

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, feat_map, tokens):
        B, C, H, W = feat_map.shape
        q = self.q_proj(feat_map).view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = self.k_proj(tokens)  # [B, L, C]
        v = self.v_proj(tokens)  # [B, L, C]
        attn = torch.softmax(q @ k.transpose(1, 2) / self.scale, dim=-1)  # [B, HW, L]
        out = (attn @ v).permute(0, 2, 1).view(B, C, H, W)
        return feat_map + out  # residual context injection


class UpsampleHead(nn.Module):
    """Upsamples the 4x-downsampled feature difference back to full resolution."""

    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, 1),
        )

    def forward(self, x):
        return self.up(x)


class TransformerChangeDetector(nn.Module):
    """Full BIT/ChangeFormer-style bitemporal transformer change detector."""

    def __init__(self, in_channels=4, feat_dim=64, num_tokens=4, depth=2, heads=4, num_classes=1):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, feat_dim)
        self.tokenizer = SemanticTokenizer(feat_dim, num_tokens)
        self.context = TransformerContext(feat_dim, depth, heads)
        self.decoder = TokenDecoder(feat_dim)
        self.dropout = nn.Dropout2d(p=0.2)
        self.head = UpsampleHead(feat_dim, num_classes)

    def forward(self, pre, post, return_features=False):
        f_pre = self.backbone(pre)
        f_post = self.backbone(post)  # same backbone weights

        t_pre = self.tokenizer(f_pre)
        t_post = self.tokenizer(f_post)
        t_pre_ctx, t_post_ctx = self.context(t_pre, t_post)

        f_pre_enriched = self.decoder(f_pre, t_pre_ctx)
        f_post_enriched = self.decoder(f_post, t_post_ctx)

        diff = torch.abs(f_pre_enriched - f_post_enriched)
        diff = self.dropout(diff)
        logits = self.head(diff)  # raw logits, shape [B, 1, H, W]
        
        if return_features:
            return logits, diff
        return logits
