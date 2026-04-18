# Bestiary

Companion entries for Part 2 of *Verified Deep Learning with Lean 4*:
a catalogue of famous neural-network architectures expressed as pure
`NetSpec` values, with no training runs attached. Each entry is meant
as a read-only fixture showing "here's what this architecture looks
like in ~20 lines of Lean."

The entries are grouped by task domain. If you've read Part 1 (the VJP
proof suite), the **Familiar territory** block is where Part 1's
primitives show up at real-world scale — every layer you see has
already been VJP'd. The later blocks pick up the new task domains
(detection, segmentation, RL) and the non-vision outliers (language,
audio, 3D, multimodal, science).

## Familiar territory — vision classifiers from Part 1's primitives

All of these are image-classification backbones built out of
conv / pool / batch-norm / residual / attention / patch-embed —
the exact layer kit VJP'd in Part 1. Reading them should feel like
Part 1 at scale.

| File | Architecture | Variants | Notes |
|------|--------------|----------|-------|
| `LeNet.lean`     | LeNet                     | LeNet-5 / LeNet-300-100 | The 1998 original; 60K params, still the template |
| `AlexNet.lean`   | AlexNet                   | canonical / tiny | 2012 ImageNet winner; restarted modern deep learning |
| `SqueezeNet.lean`| SqueezeNet                | 1.0 / 1.1 / tiny | Fire modules: squeeze-then-parallel-expand, AlexNet @ 1.25M params |
| `Inception.lean` | Inception family          | GoogLeNet (v1) / v3 / v4 | Multi-scale parallel convs; 1×1 dimension-reducer invented here |
| `Xception.lean`  | Xception                  | canonical / tiny | Extreme Inception — every conv is depthwise-separable |
| `ShuffleNet.lean`| ShuffleNet v1             | 0.5× / 1× / 2× / tiny | Mobile CNN: grouped 1×1 convs + channel shuffle |
| `ShuffleNetV2.lean` | ShuffleNet v2          | 0.5× / 1× / 1.5× / 2× / tiny | FLOPs-vs-MAC practical guidelines; channel split, no groups |
| `MobileViT.lean` | MobileViT                 | S / XS / XXS / tiny | Hybrid CNN + transformer-across-patches for mobile |
| `ConvNeXt.lean`  | ConvNeXt                  | T / S / B / L / tiny | Modernized CNN; the "can pure convs still compete" answer |
| `SwinT.lean`     | Swin Transformer          | Swin-T / Swin-S / Swin-B / tiny | Hierarchical ViT, windowed + shifted attention |

## Object detection

Localize *and* classify. Detection heads are where the linear `NetSpec`
shape starts to creak — multi-scale FPN outputs need a graph, not a
list — but the bestiary shows the single-scale view and links to the
multi-head refactor in the Limitations section below.

| File | Architecture | Variants | Notes |
|------|--------------|----------|-------|
| `YOLO.lean`      | YOLO v1/v3/v5/v8/v11      | full / fast / tiny, v3/v5s/v5m/v8n/v8s/v11 | One-shot detection; v1 raw convs+FCs, later versions layer CSP/Darknet blocks |
| `DETR.lean`      | DETR                      | R50 / R101 / tiny | Detection as set prediction; learned queries + cross-attention |
| `MaskRCNN.lean`  | Mask R-CNN                | backbone+FPN / RPN / box head / mask head / tiny | Two-stage detection + instance segmentation; the canonical reference |

## Semantic segmentation

Pixel-level labeling. Symmetric encoder/decoder with skip connections
is the recurring pattern; UNet is the canonical instance.

| File | Architecture | Variants | Notes |
|------|--------------|----------|-------|
| `UNet.lean`      | UNet                      | original / RGB / small / tiny | Encoder-decoder with skip connections; diffusion backbone |
| `SAM.lean`       | SAM (Segment Anything)    | ViT-B / L / H encoders + mask decoder + tiny | Promptable segmentation; ViT backbone does 99% of the work |
| `SegFormer.lean` | SegFormer                 | B0 / B2 / B5 + decoder + tiny | Hierarchical transformer pyramid + trivial MLP decoder |
| `DeepLabV3Plus.lean` | DeepLab v3+           | ResNet-101 / MobileNet v2 / tiny | Atrous convs + ASPP + lightweight decoder; remote-sensing workhorse |

## Image generation

Networks whose output is a novel image, not a label. The backbones
overlap heavily with segmentation (UNet shows up everywhere), but
the training objective is generative: denoising a random input until
it becomes a plausible sample from a learned distribution.

| File | Architecture | Variants | Notes |
|------|--------------|----------|-------|
| `VAE.lean`       | VAE (vanilla)             | MNIST MLP + CIFAR conv + tiny | Classical variational autoencoder; encoder outputs (μ, log σ²) |
| `DCGAN.lean`     | DCGAN                     | generator + discriminator + projector + tiny | Deep Convolutional GAN; Radford's eight guidelines made GANs trainable |
| `Pix2Pix.lean`   | Pix2Pix                   | UNet generator + PatchGAN discriminator + tiny | Paired image translation; UNet + PatchGAN + L1 loss template |
| `CycleGAN.lean`  | CycleGAN                  | generator + PatchGAN discriminator + tiny | Unpaired image translation via cycle consistency; 2 Gs + 2 Ds |
| `Diffusion.lean` | DDPM                   | CIFAR / 256 / tiny + timestep-embed | Denoising diffusion; the denoiser IS a UNet, rest is training loop |
| `StableDiffusion.lean` | Stable Diffusion 1.5 | VAE enc/dec + text enc + UNet + spatial-transformer + tiny | Latent diffusion + text cross-attention; 3 pretrained components, 1 trained |

## Reinforcement learning

Two-headed (policy + value) networks wrapped in a self-play / MCTS
outer loop. The architectural side is pure CNNs; the complexity lives
in the outer loop.

| File | Architecture | Variants | Notes |
|------|--------------|----------|-------|
| `AlphaGo.lean`   | AlphaGo (2016)            | policy + value + rollout + tiny | Three separate nets, 48 hand-crafted features; the pre-Zero original |
| `AlphaZero.lean` | AlphaZero / AlphaGo Zero  | original Go, chess, tiny | Two-headed (policy + value) |
| `MuZero.lean`    | MuZero                    | Go / Atari / tiny | AlphaZero + learned dynamics; three networks (rep + dyn + pred) |

## Beyond vision

Architectures where the task domain is not a 2D image: language,
audio, 3D scene reconstruction, multimodal embedding, scientific.
Several of these (NeRF, CLIP) have essentially *no* architectural
novelty — the interesting work lives in the data, loss, or training
procedure, and the bestiary entry exists to make that point.

| File | Architecture | Variants | Notes |
|------|--------------|----------|-------|
| `Mamba.lean`     | Mamba (selective SSM)     | 130M / 370M / 790M / tiny | Language model, linear-time alternative to attention |
| `BERT.lean`      | BERT / RoBERTa            | base / large × 2 + tiny | Encoder-only transformer; RoBERTa = BERT architecturally |
| `GPT.lean`       | GPT-1 / GPT-2             | GPT-1 + GPT-2 small/med/large/XL + tiny | Decoder-only transformer; BERT with a causal mask and tied LM head |
| `Nystromformer.lean` | Nyströmformer         | base / large + tiny | O(n) attention via 1928 Nyström trick; same params as BERT, different compute |
| `WaveNet.lean`   | WaveNet                   | speech / 3-stack / music / tiny | Dilated causal convs for audio; exponential receptive field |
| `Whisper.lean`   | Whisper                   | tiny / base / small / medium / large + decoder + tiny | Audio → text encoder-decoder transformer; multitask via token prefix |
| `NeRF.lean`      | NeRF                      | canonical / fast / tiny | 3D scene as an MLP; the magic is in positional encoding |
| `CLIP.lean`      | CLIP                      | RN50 / ViT-B/32 / ViT-L/14 / tiny | Dual encoder + contrastive; zero new primitives |
| `LLaVA.lean`     | LLaVA / LLaVA-1.5         | ViT-L + projector × 2 + LM 7B/13B + tiny | Frozen CLIP + MLP projector + LLaMA; projector is 0.3% of total |
| `Evoformer.lean` | AlphaFold 2 Evoformer     | full / mini / tiny | Dual-representation (MSA + pair) via triangle updates |

## Adding a new entry

1. Create `Bestiary/YourModel.lean`.
2. Declare the architecture as one or more `NetSpec` values (no
   `TrainConfig`, no `main` train loop).
3. Write a print-only `main` that walks through each spec and calls
   `archStr`, `totalParams`, `validate` — the pattern in `AlphaZero.lean`.
4. Register the executable in `lakefile.lean`:
   ```lean
   lean_exe «bestiary-yourmodel» where
     root := `Bestiary.YourModel
   ```
5. Run: `lake build bestiary-yourmodel && .lake/build/bin/bestiary-yourmodel`.

## Why print-only

The bestiary is for showing architecture at the conceptual level. Training
introduces dataset / loss / optimizer / GPU concerns that distract from
"here's the layer layout." A reader can always take any spec and pair it
with a `TrainConfig` to run a real training job — that's one line of code
away, same machinery as the `MainResnetTrain.lean` pattern.

## Limitations acknowledged

The current `NetSpec` is a linear list of layers, so architectures with
multiple output heads (AlphaZero, Siamese nets, multi-task models) appear
as two separate specs sharing prose-level structure rather than a real
shared-body graph. Fixing this would require either:

- A `NetSpec.branch` layer holding a list of sub-specs, or
- A free monad / effect-style spec DSL replacing the `List Layer` form.

Both are real refactors. For bestiary purposes, showing the two forks
side-by-side is both honest and pedagogically clearer than a tree that
might obscure how simple AlphaZero really is.
