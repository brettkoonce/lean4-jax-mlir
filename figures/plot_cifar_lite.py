#!/usr/bin/env python3
"""CIFAR-Lite BN vs no-BN learning curves."""
from pathlib import Path

from plotlib import plot_bn_vs_nobn

REPO = Path(__file__).resolve().parent.parent

plot_bn_vs_nobn(
    log_nobn = REPO / "logs" / "ablation_cifar-lite-nobn-sgd002.log",
    log_bn   = REPO / "logs" / "ablation_cifar-lite-bn-sgd002.log",
    out      = REPO / "figures" / "cifar_lite_bn_vs_nobn.png",
    suptitle = "CIFAR-10 · lite arch ([2]×2×[3×3] + GAP + Dense(128→10)) · SGD 0.002 + momentum",
)
