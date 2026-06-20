import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Patchify and project an image, prepending a learnable CLS token.

    A Conv2d with kernel_size == stride == patch_size performs the
    non-overlapping patch split and linear projection in one operation.

    Input:  (B, in_channels, img_size, img_size)
    Output: (B, 1 + num_patches, embed_dim)
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)              # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)              # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)         # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x