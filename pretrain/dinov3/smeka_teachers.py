"""Frozen pathology teachers used by RuiPath-Global."""

from pathlib import Path

import torch
from torch import nn


TEACHER_DIMS = {"virchow2": 1280, "uni_v2": 1536, "H-optimus-1": 1536}
TEACHER_NORMALIZATION = {
    "virchow2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "uni_v2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "H-optimus-1": ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)),
}


def build_teacher(name):
    import timm

    shared = dict(img_size=224, patch_size=14, num_classes=0)
    if name == "virchow2":
        return timm.create_model(
            "vit_huge_patch14_224", **shared, init_values=1e-5,
            mlp_ratio=5.3375, mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU, reg_tokens=4, dynamic_img_size=True, global_pool="",
        )
    if name == "uni_v2":
        return timm.create_model(
            "vit_giant_patch14_224", **shared, depth=24, num_heads=24,
            init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
            no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU, reg_tokens=8, dynamic_img_size=True,
        )
    if name == "H-optimus-1":
        return timm.create_model(
            "vit_giant_patch14_reg4_dinov2", **shared,
            embed_dim=1536, global_pool="token",
        )
    raise ValueError(f"Unsupported SMEKA teacher: {name}")


class FrozenTeachers:
    """Loaded after FSDP setup, outside the trainable model state."""

    def __init__(self, checkpoints, device):
        self.models = {}
        for name, checkpoint in checkpoints.items():
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"SMEKA {name} checkpoint not found: {path}")
            with torch.device("meta"):
                model = build_teacher(name)
            model.to_empty(device=torch.device("cuda", device))
            state = torch.load(path, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            self.models[name] = model

    @torch.no_grad()
    def __call__(self, crops):
        outputs = []
        for name, model in self.models.items():
            features = model.forward_features(crops[name].to(dtype=next(model.parameters()).dtype))
            if isinstance(features, dict):
                raise TypeError(f"{name} forward_features must return a token tensor")
            outputs.append({
                "cls_token": features[:, 0],
                "patch_tokens": features[:, model.num_prefix_tokens:],
            })
        return outputs
