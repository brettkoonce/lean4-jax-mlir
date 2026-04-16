#!/usr/bin/env python3
"""Overlay the CIFAR-Lite BN vs no-BN learning curves.

Parses train loss (per epoch) and val accuracy (at epochs 10/20/30) from
the two ablation logs and produces a 2-panel figure.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_NOBN = "ablation_cifar-lite-nobn-sgd002.log"
LOG_BN   = "ablation_cifar-lite-bn-sgd002.log"
OUT      = "figures/cifar_lite_bn_vs_nobn.png"

LOSS_RE = re.compile(r"^Epoch\s+(\d+)/\d+: loss=([\d.]+)")
VAL_RE  = re.compile(r"val accuracy.*=\s*([\d.]+)%")

def parse(path):
    losses = []  # (epoch, loss)
    vals = []    # val acc %, in order — we know they correspond to epochs 10, 20, 30
    for line in Path(path).read_text().splitlines():
        m = LOSS_RE.match(line)
        if m:
            losses.append((int(m.group(1)), float(m.group(2))))
        m = VAL_RE.search(line)
        if m:
            vals.append(float(m.group(1)))
    return losses, vals

nobn_loss, nobn_val = parse(LOG_NOBN)
bn_loss,   bn_val   = parse(LOG_BN)

val_epochs = [10, 20, 30]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: training loss per epoch
e, l = zip(*nobn_loss)
ax1.plot(e, l, label="no BN", color="#d62728", lw=2)
e, l = zip(*bn_loss)
ax1.plot(e, l, label="BN",    color="#1f77b4", lw=2)
ax1.set_xlabel("epoch")
ax1.set_ylabel("training loss")
ax1.set_title("Training loss")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right")

# Panel 2: val acc at the three measured epochs
ax2.plot(val_epochs, nobn_val, "o-", label="no BN", color="#d62728", lw=2, ms=8)
ax2.plot(val_epochs, bn_val,   "o-", label="BN",    color="#1f77b4", lw=2, ms=8)
for e, v in zip(val_epochs, nobn_val):
    ax2.annotate(f"{v:.1f}%", (e, v), xytext=(5, -14), textcoords="offset points", fontsize=9, color="#d62728")
for e, v in zip(val_epochs, bn_val):
    ax2.annotate(f"{v:.1f}%", (e, v), xytext=(5, 6),   textcoords="offset points", fontsize=9, color="#1f77b4")
ax2.set_xlabel("epoch")
ax2.set_ylabel("val accuracy (%)")
ax2.set_title("Validation accuracy")
ax2.set_xticks(val_epochs)
ax2.grid(alpha=0.3)
ax2.legend(loc="lower right")

# Shared title
fig.suptitle(
    "CIFAR-10 · lite arch ([2]×2×[3×3] + GAP + Dense(128→10)) · SGD 0.002 + momentum",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
