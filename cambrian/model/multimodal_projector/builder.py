import re

import torch.nn as nn


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class MLPConvCompressProjector(nn.Module):
    """Token compression projector: 2-layer MLP + GELU + Conv2d(k=3, s=2, p=1).

    Halves each spatial dimension, i.e. reduces the number of visual tokens by 4x
    (576 -> 144 for a 24x24 grid). Used by default for video inputs.
    """

    def __init__(self, mm_hidden_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (N, num_patches, mm_hidden_size)
        x = self.mlp(x)  # (N, num_patches, hidden_size)
        N, L, D = x.shape
        H = W = int(round(L ** 0.5))
        assert H * W == L, f"num_patches={L} is not a perfect square"
        x = x.reshape(N, H, W, D).permute(0, 3, 1, 2)  # (N, D, H, W)
        x = self.conv(x)  # (N, D, H//2, W//2)
        x = x.permute(0, 2, 3, 1).reshape(N, -1, D)  # (N, (H//2)*(W//2), D)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "mlp2x_gelu_conv2x":
        return MLPConvCompressProjector(config.mm_hidden_size, config.hidden_size)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
