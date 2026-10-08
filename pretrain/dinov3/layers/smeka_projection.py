"""Map heterogeneous teacher CLS embeddings to RuiPath register width."""

from torch import nn


class SMEKAProjection(nn.Module):
    def __init__(self, teacher_dims, student_dim):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, student_dim) for dim in teacher_dims])

    def init_weights(self):
        for head in self.heads:
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, cls_tokens):
        return [head(token.to(dtype=head.weight.dtype)) for head, token in zip(self.heads, cls_tokens)]
