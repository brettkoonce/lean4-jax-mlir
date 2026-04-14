import LeanMlir

/-! Ablation study runner. Single binary that runs any ablation config
    based on a command-line argument.

    Usage: .lake/build/bin/ablation <name> [dataDir]

    The ablation name selects a (spec, config, dataset) triple. Each
    configuration strips one component from the full recipe to measure
    its contribution.

    Results go into .lake/build/<spec_name>_<ablation>_params.bin etc.
    The spec name includes the ablation suffix so cached vmfbs don't
    collide between runs. -/

-- ═══════════════════════════════════════════════════════════════════
-- Specs
-- ═══════════════════════════════════════════════════════════════════

-- Chapter 1: MLP on MNIST (no convolutions)
def mnistMlp : NetSpec where
  name := "MNIST-MLP"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

-- Chapter 2: CNN on MNIST (S4TF book match — no BN, 2 conv layers)
def mnistCnnNoBn : NetSpec where
  name := "MNIST-CNN-noBN"
  imageH := 28
  imageW := 28
  layers := [
    .conv2d 1 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 6272 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

-- Chapter 2 variant: CNN with BN (our architecture)
def mnistCnnBn : NetSpec where
  name := "MNIST-CNN-BN"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 3136 512 .relu,
    .dense 512 10 .identity
  ]

-- Chapter 3: CNN on CIFAR (no BN)
def cifarCnnNoBn : NetSpec where
  name := "CIFAR-CNN-noBN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d 3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

-- Chapter 3: CNN on CIFAR (with BN)
def cifarCnnBn : NetSpec where
  name := "CIFAR-CNN-BN"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

-- ═══════════════════════════════════════════════════════════════════
-- Configs (each ablation strips one thing from the full recipe)
-- ═══════════════════════════════════════════════════════════════════

-- Full recipe for small models
def fullRecipe (lr : Float := 0.001) (epochs : Nat := 15) (batch : Nat := 128) : TrainConfig where
  learningRate := lr
  batchSize    := batch
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 1
  augment      := true
  labelSmoothing := 0.1

-- S4TF book baseline: SGD 0.1, constant LR, no regularization
def s4tfBaseline (epochs : Nat := 12) : TrainConfig where
  learningRate := 0.1
  batchSize    := 128
  epochs       := epochs
  useAdam      := false
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- SGD with lower LR (0.01)
def sgdLowLr (epochs : Nat := 15) : TrainConfig where
  learningRate := 0.01
  batchSize    := 128
  epochs       := epochs
  useAdam      := false
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- Adam only (no aug, no smooth, no wd, no cosine)
def adamOnly (epochs : Nat := 15) : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- Adam + cosine (no aug, no smooth, no wd)
def adamCosine (epochs : Nat := 15) : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false
  labelSmoothing := 0.0

-- Adam + cosine + augmentation (no smooth, no wd)
def adamCosineAug (epochs : Nat := 30) : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 2
  augment      := true
  labelSmoothing := 0.0

-- ═══════════════════════════════════════════════════════════════════
-- Ablation registry
-- ═══════════════════════════════════════════════════════════════════

structure AblationRun where
  spec     : NetSpec
  config   : TrainConfig
  dataset  : DatasetKind
  dataDir  : String  -- default data dir

def ablations : List (String × AblationRun) := [
  -- Chapter 1: MLP on MNIST
  ("mlp-sgd",         ⟨mnistMlp,     s4tfBaseline 12,  .mnist, "data"⟩),
  ("mlp-sgd-low",     ⟨mnistMlp,     sgdLowLr 15,     .mnist, "data"⟩),
  ("mlp-adam",         ⟨mnistMlp,     adamOnly 12,     .mnist, "data"⟩),
  ("mlp-full",         ⟨mnistMlp,     fullRecipe 0.001 12 128, .mnist, "data"⟩),

  -- Chapter 2: CNN on MNIST (no BN, S4TF architecture)
  ("cnn-nobn-sgd",     ⟨mnistCnnNoBn, s4tfBaseline 12, .mnist, "data"⟩),
  ("cnn-nobn-adam",    ⟨mnistCnnNoBn, adamOnly 15,     .mnist, "data"⟩),

  -- Chapter 2: CNN on MNIST (with BN, our architecture)
  ("cnn-bn-sgd",       ⟨mnistCnnBn,   s4tfBaseline 15, .mnist, "data"⟩),
  ("cnn-bn-adam",      ⟨mnistCnnBn,   adamOnly 15,     .mnist, "data"⟩),
  ("cnn-bn-full",      ⟨mnistCnnBn,   fullRecipe 0.001 15 128, .mnist, "data"⟩),

  -- Chapter 3: CIFAR (no BN — should struggle)
  ("cifar-nobn-sgd",   ⟨cifarCnnNoBn, s4tfBaseline 30, .cifar10, "data"⟩),
  ("cifar-nobn-adam",  ⟨cifarCnnNoBn, adamOnly 30,     .cifar10, "data"⟩),

  -- Chapter 3: CIFAR (with BN — the unlock)
  ("cifar-bn-sgd",     ⟨cifarCnnBn,   s4tfBaseline 30, .cifar10, "data"⟩),
  ("cifar-bn-adam",    ⟨cifarCnnBn,   adamOnly 30,     .cifar10, "data"⟩),
  ("cifar-bn-cosine",  ⟨cifarCnnBn,   adamCosine 30,   .cifar10, "data"⟩),
  ("cifar-bn-aug",     ⟨cifarCnnBn,   adamCosineAug 30, .cifar10, "data"⟩),
  ("cifar-bn-full",    ⟨cifarCnnBn,   fullRecipe 0.001 30 128, .cifar10, "data"⟩)
]

def main (args : List String) : IO Unit := do
  let name := args.head?.getD ""
  if name == "" || name == "--list" then
    IO.println "Available ablations:"
    for (n, _) in ablations do
      IO.println s!"  {n}"
    IO.println "\nUsage: ablation <name> [dataDir]"
    return

  match ablations.lookup name with
  | some run =>
    let dataDir := (args.tail.head?).getD run.dataDir
    -- Override the spec name to include the ablation suffix for unique vmfb paths
    let spec := { run.spec with name := run.spec.name ++ "-" ++ name }
    IO.eprintln s!"Ablation: {name}"
    IO.eprintln s!"  spec: {run.spec.name}, optimizer: {if run.config.useAdam then "Adam" else "SGD"}"
    IO.eprintln s!"  lr: {run.config.learningRate}, cosine: {run.config.cosineDecay}, wd: {run.config.weightDecay}"
    IO.eprintln s!"  aug: {run.config.augment}, label_smooth: {run.config.labelSmoothing}"
    spec.train run.config dataDir run.dataset
  | none =>
    IO.eprintln s!"Unknown ablation: {name}"
    IO.eprintln "Use --list to see available ablations."
    IO.Process.exit 1
