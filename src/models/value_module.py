"""Value module: V_k = W_y(label_dist_k) + value_k  (decision 2).

Each prototype carries (a) a learnable label representation and (b) a learnable
value vector. The label part keeps prototypes interpretable ("this prototype
represents class c / regression level v"), which is central to our interpretability
claim (EXPERIMENT_PLAN.md §4 decision 2).

Classification: label_dist_k is a (K, C) logit table -> softmax -> embed via W_y (C->d).
Regression:     label_level_k is a (K, 1) learnable scalar -> Linear(1->d).

Requires PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueModule(nn.Module):
    def __init__(self, n_prototypes: int, dim: int, task: str, n_classes: int = 2):
        """
        Args:
            n_prototypes: K.
            dim: representation dim d.
            task: "binclass" | "multiclass" | "regression".
            n_classes: number of classes (classification only).
        """
        super().__init__()
        self.K = n_prototypes
        self.d = dim
        self.task = task.lower()

        self.value = nn.Parameter(torch.zeros(n_prototypes, dim))   # value_k
        nn.init.normal_(self.value, std=0.02)

        if self.task in {"binclass", "multiclass"}:
            C = n_classes
            self.label_logits = nn.Parameter(torch.zeros(n_prototypes, C))  # per-proto class dist
            nn.init.normal_(self.label_logits, std=0.02)
            self.W_y = nn.Linear(C, dim, bias=False)                        # W_y: C -> d
            self.n_classes = C
        elif self.task == "regression":
            self.label_level = nn.Parameter(torch.zeros(n_prototypes, 1))   # per-proto target level
            nn.init.normal_(self.label_level, std=0.02)
            self.W_y = nn.Linear(1, dim, bias=False)
        else:
            raise ValueError(f"Unknown task: {task}")

    def values(self) -> torch.Tensor:
        """Return V_k for all prototypes: (K, d)."""
        if self.task in {"binclass", "multiclass"}:
            label_emb = self.W_y(F.softmax(self.label_logits, dim=-1))      # (K, d)
        else:
            label_emb = self.W_y(self.label_level)                          # (K, d)
        return label_emb + self.value                                       # (K, d)

    def prototype_label_summary(self) -> torch.Tensor:
        """For interpretability: the learned per-prototype label representation.

        Classification -> (K, C) class probabilities; Regression -> (K,) levels.
        Used in Phase-1 Test 3 / Phase-2 case studies (what each prototype means).
        """
        if self.task in {"binclass", "multiclass"}:
            return F.softmax(self.label_logits, dim=-1).detach()
        return self.label_level.squeeze(-1).detach()
