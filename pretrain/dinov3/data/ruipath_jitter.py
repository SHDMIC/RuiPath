import math

import torch
from torch.nn import functional as F

EPSILON = 1e-9


class RuiPathJitter:
    """OD-space stain perturbation"""

    def __init__(
        self,
        sigma: float = 0.15,
        iters: int = 100,
        lr: float = 5e-3,
        mean=(0.637, 0.464, 0.602),
        std=(0.21, 0.196, 0.184),
    ):
        self.device = None
        self.sigma = sigma
        self.iters = iters
        self.lr = lr
        # Central 95% interval of LogNormal(0, sigma^2)
        self.alpha_lo = math.exp(-1.96 * sigma)
        self.alpha_hi = math.exp(1.96 * sigma)
        # Keep plain values here; tensors are materialized lazily in .to()
        # (the model is built under the meta-device context of FSDP2, where
        # tensor creation in __init__ would produce meta tensors).
        self.mean = tuple(mean)
        self.std = tuple(std)
        # Reference H&E stain basis in OD space, columns: [Hematoxylin, Eosin]
        self.stain_matrix_init = (
            (0.71, 0.0),  # R
            (0.65, 0.69),  # G
            (0.27, 0.724),  # B
        )

    def to(self, device):
        if self.device is None:
            self.device = device
            self._mean = torch.tensor(self.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            self._std = torch.tensor(self.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            self._init_stain = torch.tensor(self.stain_matrix_init, dtype=torch.float32, device=device)

    def sample_scale(self, batch_size: int) -> torch.Tensor:
        """Log-normal scaling factors with reset-to-identity outside the
        central 95% interval. Returns a [B, 2, 1] tensor (H and E channels)."""
        alpha = torch.exp(self.sigma * torch.randn(batch_size, 2, 1, device=self.device))
        outside = (alpha < self.alpha_lo) | (alpha > self.alpha_hi)
        return torch.where(outside, torch.ones_like(alpha), alpha)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """image: [B, 3, H, W] tensor normalized with (mean, std)."""
        batch_size, channel_size, height, width = image.shape
        image_dtype = image.dtype

        # De-normalize to raw RGB in [0, 255]
        rgb = ((image.float() * self._std + self._mean) * 255.0).clamp_(0.0, 255.0)
        # [B, 3, P]
        image_od = -torch.log10(rgb / 255.0 + EPSILON).reshape(batch_size, channel_size, -1)

        # Stain estimation: least squares + optional Vahadane-style refinement
        stain_matrix = self._init_stain.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 2]
        with torch.no_grad():
            od_tensor = torch.linalg.lstsq(stain_matrix, image_od).solution.clamp_(min=EPSILON)  # [B, 2, P]
        if self.iters > 0:
            stain_matrix = stain_matrix.requires_grad_(True)
            od_tensor = od_tensor.clone().requires_grad_(True)
            optimizer = torch.optim.AdamW([stain_matrix, od_tensor], lr=self.lr, weight_decay=0)
            for _ in range(self.iters):
                optimizer.zero_grad()
                loss = F.mse_loss(torch.bmm(stain_matrix, od_tensor), image_od)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    stain_matrix.clamp_(min=EPSILON)
                    od_tensor.clamp_(min=EPSILON)
                    # Unit-norm columns, as in Vahadane et al.
                    stain_matrix.div_(stain_matrix.norm(p=2, dim=1, keepdim=True).clamp_(min=EPSILON))
            stain_matrix = stain_matrix.detach()
            od_tensor = od_tensor.detach()

        # Log-normal channel-wise perturbation of the stain concentrations
        scale = self.sample_scale(batch_size)  # [B, 2, 1]
        od_jittered = torch.bmm(stain_matrix, od_tensor * scale)  # [B, 3, P]
        rgb_jittered = (255.0 * torch.pow(10.0, -od_jittered)).clamp_(0.0, 255.0)
        rgb_jittered = rgb_jittered.reshape(batch_size, channel_size, height, width)

        # Re-normalize to the training pipeline's input space
        out = (rgb_jittered / 255.0 - self._mean) / self._std
        return out.to(dtype=image_dtype)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sigma={self.sigma}, iters={self.iters}, lr={self.lr}, "
            f"alpha_range=({self.alpha_lo:.3f}, {self.alpha_hi:.3f}))"
        )
