"""Relational multi-expert distillation from the RuiPath supplementary methods."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SMEKALoss(nn.Module):
    def __init__(self, temperature=0.1, p2p_weight=1.0, c2p_weight=1.0, register_weight=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("SMEKA temperature must be positive")
        self.temperature = temperature
        self.p2p_weight = p2p_weight
        self.c2p_weight = c2p_weight
        self.register_weight = register_weight

    def _p2p_logits(self, patches):
        patches = F.normalize(patches.float(), dim=-1)
        return F.relu(patches @ patches.transpose(-1, -2)) / self.temperature

    def _c2p_logits(self, cls, patches):
        cls = F.normalize(cls.float(), dim=-1)
        patches = F.normalize(patches.float(), dim=-1)
        return (cls.unsqueeze(1) @ patches.transpose(-1, -2)).squeeze(1) / self.temperature

    def entropy_weights(self, teacher_p2p_logits):
        entropies = []
        for logits in teacher_p2p_logits:
            count = logits.shape[-1]
            if count < 2:
                raise ValueError("P2P distillation requires at least two patches")
            log_prob = logits.float().log_softmax(dim=-1)
            prob = log_prob.exp()
            entropy = -(prob * log_prob).sum(dim=-1).mean(dim=-1) / math.log(count)
            entropies.append(entropy)
        return (-torch.stack(entropies, dim=1)).softmax(dim=1)

    @staticmethod
    def _kl(student_logits, teacher_logits):
        loss = F.kl_div(
            student_logits.float().log_softmax(dim=-1),
            teacher_logits.float().softmax(dim=-1),
            reduction="none",
        ).sum(dim=-1)
        return loss.mean(dim=-1) if loss.ndim == 2 else loss

    def forward(self, student, teachers, projected_teacher_cls):
        if len(teachers) != len(projected_teacher_cls) or len(teachers) > student["reg_tokens"].shape[1]:
            raise ValueError("Provide one register slot and projected CLS per teacher")
        student_patches = student["patch_tokens"]
        student_cls = student["cls_token"]
        teacher_logits = [self._p2p_logits(t["patch_tokens"]) for t in teachers]
        weights = self.entropy_weights(teacher_logits).detach()
        terms = {"p2p": [], "c2p": [], "register": []}
        for index, teacher in enumerate(teachers):
            count = teacher["patch_tokens"].shape[1]
            side = math.isqrt(count)
            student_side = math.isqrt(student_patches.shape[1])
            if side * side != count or student_side * student_side != student_patches.shape[1]:
                raise ValueError("Patch grids must be square for spatial pooling")
            pooled = F.adaptive_avg_pool2d(
                student_patches.transpose(1, 2).reshape(student_patches.shape[0], -1, student_side, student_side),
                (side, side),
            ).flatten(2).transpose(1, 2)
            terms["p2p"].append(self._kl(self._p2p_logits(pooled), teacher_logits[index]))
            terms["c2p"].append(self._kl(
                self._c2p_logits(student_cls, pooled),
                self._c2p_logits(teacher["cls_token"], teacher["patch_tokens"]),
            ))
            terms["register"].append(self._kl(
                student["reg_tokens"][:, index].float() / self.temperature,
                projected_teacher_cls[index].float() / self.temperature,
            ))
        reduced = {key: (torch.stack(values, dim=1) * weights).sum(dim=1).mean() for key, values in terms.items()}
        total = (self.p2p_weight * reduced["p2p"] + self.c2p_weight * reduced["c2p"]
                 + self.register_weight * reduced["register"])
        return total, reduced
