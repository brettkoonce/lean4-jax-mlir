# Side quests

A grab bag of optional extensions to the codegen, training pipeline, and ablation toolkit. Nothing here is on the primary chapter path — the book's spine works without any of them. They're listed because someone in a "weird domain" (medical imaging, satellite, microscopy, audio spectrograms, anomaly detection on industrial sensors, ...) might want to pick one up and find it already most-of-the-way assembled.

Each entry has: short description, where in the codebase it slots, rough effort, and which domain typically cares.

The categories below are sorted by how invasive the change is. **Training-loop** knobs are easiest (no codegen, just `Train.lean`). **Dataloader** knobs are next (C kernel + binding). **Codegen** knobs touch the MLIR emitter. **Spec primitives** add new `Layer` constructors.

---

## Training-loop knobs (no codegen, ~1–2 hr each)

These all live in `Train.lean` + `TrainConfig` + maybe a small `F32Array` helper. None touch the MLIR emitter.

- **EMA (Exponential Moving Average of weights).** Per-step `ema_θ = β·ema_θ + (1−β)·θ`. Use `ema_θ` for eval. Cheap (one extra `F32.ema` call per step). DeiT default β=0.9999. Pairs with mixup. **Domains:** any production-deployment scenario, where ~0.2 pt of free accuracy from smoother weights is worth the trivial extra storage.

- **SWA (Stochastic Weight Averaging).** Equal-weight average of the last K epochs of training. After the cosine-decay tail flattens out, swap LR to constant or cyclical and accumulate weight checkpoints. Replace eval params with the SWA average. ~+0.3 pts typical. **Domains:** anywhere training is expensive enough that you want the most-of-an-ensemble for the cost of one model.

- **SWAG (Bayesian uncertainty via SWA + sample covariance).** Maddox et al. 2019. Extends SWA: also accumulate diagonal variance + a low-rank (last K=20) deviation matrix. At test time, sample weights from `N(θ_SWA, Σ)`, run K forwards, take prediction mean+variance. K test-time samples ≈ K-fold inference cost. **Domains:** medical imaging, drug discovery, autonomous vehicles, anywhere you want calibrated uncertainty without retraining 5 models.

- **Gradient clipping (global-norm or per-tensor).** Cap gradient norm at `max_norm` before optimizer step. Stabilizes training of deep transformers and RNNs. ~10 lines in `Train.lean` + 1 FFI helper. **Domains:** transformer training, especially with mixup (which produces noisier gradients).

- **SAM (Sharpness-Aware Minimization).** Foret et al. 2020. Per-step does an *extra* forward+backward at θ + ε·grad/||grad||, then takes the gradient at that perturbed point as the actual update. ~2× the per-step compute, but consistently +0.3–1.0 pt across architectures. **Domains:** any setting where you can afford 2× train compute for a calibrated boost — competitions, leaderboard chasing, last-mile-of-deployment optimization.

- **Lookahead optimizer (Zhang et al. 2019).** Outer optimizer averages every K inner steps: `θ_slow := θ_slow + α(θ_fast − θ_slow)`. Wraps any inner optimizer. ~30 lines. **Domains:** when Adam/SGD plateau, this often gives a free 0.1–0.3 pt with no hyperparameter tuning.

- **Layer-wise LR decay (LLRD).** Different LR per layer (typically lower for early layers). ~1 line in the LR schedule for vision-transformer fine-tuning. **Domains:** transfer learning, fine-tuning a pretrained ViT on a small downstream dataset.

- **Snapshot ensembles.** Cyclical LR + save N checkpoints at LR minima → ensemble at inference. Free if you train cyclical anyway. **Domains:** medical / scientific where ensemble accuracy is worth N× inference cost.

---

## Dataloader-only augmentations (no codegen, ~30 min – 4 hr each)

C kernel in `ffi/f32_helpers.c` + Lean binding in `F32Array.lean` + 1 line in `Train.lean`. The mixup/cutmix/RE pack we already shipped is the template.

- **RandAugment (Cubuk et al. 2020).** Pick `n=2` random transforms per image from a pool of 14 (rotate, shear-x/y, translate, color, brightness, contrast, sharpness, posterize, solarize, equalize, autocontrast, invert, identity), each with a global magnitude `m=9`. Most-cited single augmentation in modern recipes after mixup/cutmix. ~3 hr code. **Domains:** any image task where you have <1M train images.

- **AugMix (Hendrycks et al. 2019).** Sum of 3 random augmentation chains, each chain 1–3 ops deep, weighted by a Dirichlet sample. Different distribution-shift profile than RandAugment — better for OOD robustness. ~2 hr. **Domains:** anywhere distribution shift between train and test is a concern (medical imaging across hospitals, satellite imagery across seasons).

- **TrivialAugment (Müller & Hutter 2021).** RandAugment but with `n=1` and uniform-random magnitude per call. Simpler, often equals RandAugment. ~1 hr. **Domains:** "I want to add aug but don't want to tune n,m".

- **Cutout / Random Erasing variants.** Already shipped Random Erasing. Cutout is the simpler "always erase one fixed-size square per image" variant. **Domains:** small datasets where you want to fight overfitting.

- **TTA (Test-Time Augmentation).** N augmented copies per test image, average the predictions. Free at training time, costs N× at eval. ~50 lines. **Domains:** medical imaging where one mistake is expensive enough that 10× inference cost is fine.

- **Class-balanced sampling / weighted sampling.** Use `W ∝ 1/n_class` to oversample rare classes. ~30 lines in dataloader. **Domains:** medical imaging (rare conditions), anomaly detection, fraud detection.

- **MixCut / SaliencyMix / PuzzleMix.** Variants of CutMix that sample paste regions based on attention/saliency rather than uniformly. More compute, sometimes better. ~2–4 hr per variant. **Domains:** competition territory.

- **Repeated augmentation.** Sample K augmentations of *each* image into the batch (so effective batch = N × K with N unique images). DeiT used K=3. Improves convergence at the same wall-clock cost. ~30 lines in batch sampler. **Domains:** transformer training.

---

## Codegen / MLIR-emit knobs (modify `MlirCodegen.lean`)

These touch the StableHLO emitter and the matching backward. Each is a one-shot lift but spans every architecture using the affected op.

- **Stochastic Depth (Huang et al. 2016).** Per-block residual drop with probability `p_k = k/N · p_max`. Forward: `out = a + α·f(a)`, backward: `da_inner = α·dy`. Affects every residual fan-in across `residualBlock`, `bottleneckBlock`, `mbConv`, `mbConvV3`, `fusedMbConv`, `uib`, `convNextStage`, `transformerEncoder`. New `[N]` alpha tensor passed at train time, sampled in C, eval signature unchanged (or rescales f). ~half day for the sweep across all block types. **Domains:** deep transformers, any residual stack >20 blocks. DeiT default for ViT-S.

- **Dropout (forward + backward).** Insert `mask · (x / (1−p))` in training, identity at eval. Needs RNG state plumbing in MLIR. Touch points: dense layers, attention output projection, MLP block. ~3–4 hr. Not popular for vision (mixup substitutes) but standard for NLP and many bioinformatics models.

- **DropPath / DropBlock.** Spatial-coherent dropout (drops contiguous channels or 2D regions instead of independent pixels). DropBlock is more aggressive than DropPath. ~half day. **Domains:** segmentation, dense-prediction tasks.

- **Mixed precision (fp16/bf16).** Emit fp16 forward, fp32 master weights, fp16 backward. Halves activation memory and ~2× throughput on supported hardware. Touches every op in `MlirCodegen.lean` for type promotion. Real architectural lift — ~1–2 weeks. **Domains:** anyone with ConvNeXt-T or larger trying to fit in 8–24 GB.

- **Gradient checkpointing.** Recompute activations during backward instead of storing them. Trades ~30% extra forward time for ~50% activation memory savings. Codegen change: insert `recompute` markers, modify backward to re-run forward chunks. ~1 week. **Domains:** training larger models on smaller GPUs, e.g. ViT-B on a 16 GB card.

- **Heteroscedastic regression head.** Replace `dense N→K .identity` with two heads: `dense N→K` for μ and `dense N→K` for log-variance. Train with NLL `0.5·(y−μ)²·exp(−log_σ²) + 0.5·log_σ²`. ~30 lines codegen + new loss. **Domains:** medical regression (e.g., Bayesian dose estimation), age estimation, distance estimation.

- **Manifold Mixup (Verma et al. 2019).** Mix at a *hidden* layer rather than the input. Pick a random layer, do mixup at its activations, propagate the mixed-label through the rest. Codegen: add a "mixup point" branch in the train-time forward emit. ~2–3 hr. **Domains:** less common than input mixup, but stronger regularizer when it works.

- **Adversarial training (PGD/FGSM).** Per-step: forward, backward → δ ∝ sign(∇x), clip, forward again at x+δ, backward, then optimizer step. ~2× compute. Codegen needs an extra forward+backward in the train-step body. ~half day. **Domains:** safety-critical (medical, autonomous), adversarial-robustness benchmarks.

---

## New spec primitives (add `Layer` constructor)

Each adds a new `inductive Layer` case + the matching codegen forward/backward + a Bestiary entry.

- **GroupNorm.** Like LayerNorm but groups channels. Bridges BN ↔ LN. ~half day, mostly mechanical (the channel-axis math is the same as LN-NCHW we already have, just over groups of channels). **Domains:** small-batch settings (medical, video) where BN's batch-stats are unreliable but LN's per-spatial isn't quite right.

- **InstanceNorm.** Per-image, per-channel normalization (no batch averaging). 5-line variant of BN with batch=1 axis convention. **Domains:** style transfer, image-to-image translation, generative models.

- **Multi-scale features / FPN integration.** Already have `fpnModule` as a Bestiary primitive but no codegen. Adding the codegen would unlock object detection / segmentation backbones built on top of any of our existing classifiers. ~1 day. **Domains:** detection, segmentation, anomaly localization.

- **Cross-attention block.** Like `transformerEncoder` but with separate Q (from one stream) and K, V (from another). Already half-implemented in `transformerDecoder` Bestiary entry. ~half day to wire codegen. **Domains:** image-text matching (CLIP), DETR-style detection, prompted segmentation (SAM).

- **Causal-masked attention.** Apply a lower-triangular mask before softmax. Trivial as a forward variant; backward is the same as standard attention. ~1 hr. **Domains:** autoregressive models (language, audio, time-series).

- **2D positional encoding (sinusoidal or learned).** Already have learned positional embedding in patchEmbed; sinusoidal would be a parameter-free variant. ~1 hr. **Domains:** any ViT-style spatial model, especially where you want OOD-resolution generalization.

---

## Inference / post-training extensions

- **MC Dropout (epistemic uncertainty without retraining).** Keep dropout active at inference, run K forwards, take variance of predictions. Requires the dropout codegen knob above. ~2 hr once dropout exists. **Domains:** medical, scientific.

- **Temperature scaling (calibration).** Fit a single scalar `T` on a held-out set such that `softmax(logits / T)` produces calibrated probabilities. ~50 lines, no model changes. **Domains:** anywhere you need probabilistic outputs (medical decision support, risk estimation).

- **Knowledge distillation (training new student with teacher's soft targets).** Replace hard labels in the loss with `T·logits_teacher`, optionally combined with hard labels via `α·CE_soft + (1−α)·CE_hard`. ~2 hr (mostly: a teacher-forward pass per batch). **Domains:** model compression for deployment, semi-supervised learning when teacher is large.

- **Influence functions.** For each prediction, identify the K training samples that most affected it. Computes `H⁻¹ ∇_θ L(test)` projected onto per-train-sample gradients. ~1 day. **Domains:** medical (which training cases drove this diagnosis?), data quality auditing, fairness analysis.

- **GradCAM / saliency maps.** Visualize *which* spatial region drove a prediction, by gradient of the target class w.r.t. last conv-layer activations. ~half day. **Domains:** medical imaging interpretability, debugging of any vision classifier.

---

## Loss / objective modifications

- **Focal loss (Lin et al. 2017).** `(1−p_t)^γ · CE(p, y)` — down-weights easy examples. ~30 lines codegen. **Domains:** detection, severe class imbalance, anomaly detection.

- **Label distribution learning / soft labels with temperature.** Generalizes label smoothing. **Domains:** age estimation, fine-grained classification with hierarchical labels.

- **Contrastive losses (SimCLR / BYOL / MoCo).** Pretrain self-supervised, then fine-tune. Needs a Siamese pair pipeline + a contrastive head. ~1 week. **Domains:** medical imaging where labeled data is scarce but unlabeled images are abundant.

- **Auxiliary / deep-supervision losses.** Add intermediate classifier heads at multiple depths. ~half day codegen. **Domains:** segmentation backbones, training stability for deep networks.

---

## How to pick

- **"I'm tuning a known-good architecture for accuracy"**: EMA, mixup-already-shipped, cutmix-already-shipped, SWA → SWAG.
- **"I need uncertainty estimates"**: SWAG, MC Dropout, heteroscedastic head, temperature scaling for calibration.
- **"My dataset is small / imbalanced"**: focal loss, class-balanced sampling, cutmix, RandAugment.
- **"I'm deploying and need explanations"**: GradCAM, influence functions, calibration.
- **"I'm fitting a bigger model on this hardware"**: gradient checkpointing, mixed precision.
- **"My dataset is huge but unlabeled"**: contrastive pretraining, then fine-tune.
- **"I want competition-grade results"**: SAM, snapshot ensembles, TTA, full DeiT recipe.

The pattern across all of these: each one slots into our existing 3-layer stack (`Layer` spec → MLIR codegen → Train.lean) at exactly one point. Adding any single trick is small. Adding all of them in a coherent recipe (e.g., the full DeiT training scheme) takes roughly the same total work as we've already done across Chs 2–10.
