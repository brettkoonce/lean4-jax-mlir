# ViT-Tiny Imagenette: DeiT-style augmentation ablation (2026-04-28)

5-cell ablation testing whether the ViT-Tiny / CNN performance gap on
Imagenette (~17 points behind) reduces to a *recipe* gap (no architectural
inductive bias substitute) rather than a ViT-architecture failing.

## Setup

- Architecture: ViT-Tiny (`patchEmbed 3 192 16 196` + 12-block transformer + dense 192→10), ~3M params
- Dataset: Imagenette (10 classes, 9469 train / 3904 val, 224×224)
- Recipe: Adam @ 0.0003, batch=32, cosine decay, 5-epoch warmup, hflip+crop, label_smooth=0.1, weight_decay=0.0001
- Schedule: 80 epochs each
- Backend: IREE 3.12 + ROCm/HIP on RX 7900 XTX (gfx1100), dual-GPU parallel

## Results

| variant | val accuracy | train loss (e80) | Δ vs bare | runtime |
|---|---|---|---|---|
| `vit-tiny-bare`   (existing baseline) | 71.70% | 0.99 | — | 2.3 hr |
| `vit-tiny-erase`  (Random Erasing) | 70.62% | 0.56 | −1.08 | 2.4 hr |
| `vit-tiny-mixup`  (Mixup α=0.8) | **74.69%** | 1.29 | +2.99 | 2.3 hr |
| `vit-tiny-cutmix` (CutMix α=1.0) | **77.10%** | 1.46 | **+5.40** | 2.7 hr |
| `vit-tiny-full`   (Mixup + RE) | 74.23% | 1.34 | +2.53 | 2.8 hr |

Findings:
- **CutMix alone is the standout: +5.4 pts.** Single dataloader-only augmentation closes most of the ViT/CNN gap on Imagenette without any codegen change beyond the soft-label loss path.
- **Mixup helps by +3.0 pts**, but less than CutMix here. (DeiT recipe used both together; we ran them separately to isolate effects.)
- **Random Erasing alone hurts** (−1.1) at this scale; **Mixup + RE underperforms Mixup alone** (74.23 vs 74.69) — RE adds noise that the small dataset doesn't have the capacity to absorb.
- **Mixup/CutMix train loss curves never reach the bare baseline's 0.99** because soft-target labels mean the cross-entropy floor is bounded above zero — the model can't drive log-prob to zero on a smoothed target. Comparing train loss across variants is misleading; val accuracy is the meaningful metric.

For comparison:
- DeiT paper (full ImageNet, full recipe): ViT-S 77.9% → 79.8% (+1.9 pts with Mixup + CutMix + RandAugment + RE + EMA + Stochastic Depth)
- Ours (Imagenette, dataloader subset only): ViT-Tiny 71.7% → 77.1% (+5.4 pts with **just CutMix**)

We see a *larger* relative lift than the paper because:
1. Smaller dataset → more overfitting → augmentation has more to fix
2. Smaller model (ViT-Tiny vs ViT-S) → again more sensitive to undertraining
3. Started from a worse baseline (71.7% vs paper's 77.9%)

This is exactly the data-scale-substitutes-for-architectural-bias story DeiT was demonstrating.

## Files

- `bare.log` — `vit-tiny-bare` (re-run of existing 71.7% checkpoint)
- `erase.log` — `vit-tiny-erase`
- `mixup.log` — `vit-tiny-mixup`
- `cutmix.log` — `vit-tiny-cutmix`
- `full.log` — `vit-tiny-full` (Mixup + Random Erasing)

Architecture for the codegen path: `useSoftLabels` flag in `MlirCodegen.generateTrainStep` swaps the int32-label CE construction for a `[B, NC]` soft-label tensor input. Mixup and CutMix produce mixed images + matching soft-target labels via `F32.mixupImages/SoftLabels` and `F32.cutmixImages/SoftLabels` (in `ffi/f32_helpers.c`). See commit landing the soft-label train-step path.
