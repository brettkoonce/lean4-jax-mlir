import Lake
open Lake DSL

package «lean4-jax» where
  version := v!"0.1.0"
  buildType := .release

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"

lean_lib «LeanMlir» where
  roots := #[`LeanMlir]

-- IREE FFI shim: Lean ↔ C bridge for libiree_ffi.so (see ffi/).
target ireeLeanFfiO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "iree_lean_ffi.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "iree_lean_ffi.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString,
                    "-I", (pkg.dir / "ffi").toString]
  let traceArgs := #["-fPIC", "-O2"]
  buildO oFile srcJob weakArgs traceArgs

-- F32 ByteArray helpers (He init, argmax, data loading — all in C for speed).
target f32HelpersO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "f32_helpers.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "f32_helpers.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString]
  let traceArgs := #["-fPIC", "-O2"]
  buildO oFile srcJob weakArgs traceArgs

extern_lib libireeffi pkg := do
  let shimO ← fetch <| pkg.target ``ireeLeanFfiO
  let f32O  ← fetch <| pkg.target ``f32HelpersO
  buildStaticLib (pkg.staticLibDir / nameToStaticLib "ireeffi") #[shimO, f32O]

-- ═══════════════════════════════════════════════════════════════════
-- Phase 3 trainers (Lean → MLIR → IREE → GPU)
-- ═══════════════════════════════════════════════════════════════════

private def ireeLink : Array String :=
  #["-L", "./ffi", "-liree_ffi", "-Wl,-rpath,./ffi", "-Wl,--allow-shlib-undefined"]

lean_exe «resnet34-train» where
  root := `MainResnetTrain
  moreLinkArgs := ireeLink

lean_exe «resnet50-train» where
  root := `MainResnet50Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v2-train» where
  root := `MainMobilenetV2Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v3-train» where
  root := `MainMobilenetV3Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v4-train» where
  root := `MainMobilenetV4Train
  moreLinkArgs := ireeLink

lean_exe «efficientnet-train» where
  root := `MainEfficientNetTrain
  moreLinkArgs := ireeLink

lean_exe «efficientnet-v2-train» where
  root := `MainEfficientNetV2Train
  moreLinkArgs := ireeLink

lean_exe «vit-tiny-train» where
  root := `MainVitTrain
  moreLinkArgs := ireeLink

lean_exe «ablation» where
  root := `MainAblation
  moreLinkArgs := ireeLink

lean_exe «vgg-train» where
  root := `MainVggTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-cnn-train» where
  root := `MainMnistCnnTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-bn-train» where
  root := `MainCifarCnnBnTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-mlp-train-f32» where
  root := `MainMlpTrainF32
  moreLinkArgs := ireeLink

lean_exe «cifar-cnn-train-f32» where
  root := `MainCifarTrainF32
  moreLinkArgs := ireeLink

-- ═══════════════════════════════════════════════════════════════════
-- Tests + benchmarks
-- ═══════════════════════════════════════════════════════════════════

lean_exe «test-forward» where
  root := `TestForward
  moreLinkArgs := ireeLink

lean_exe «test-iree» where
  root := `TestIreeRuntime
  moreLinkArgs := ireeLink

lean_exe «test-train» where
  root := `TestTrainStep
  moreLinkArgs := ireeLink

lean_exe «test-iree-load» where
  root := `TestIreeLoad
  moreLinkArgs := ireeLink

lean_exe «test-f32» where
  root := `TestF32
  moreLinkArgs := ireeLink

lean_exe «bench-resnet» where
  root := `BenchResnet
  moreLinkArgs := ireeLink

lean_exe «test-resnet-fwd» where
  root := `TestResnetForward

lean_exe «test-resnet-residual» where
  root := `TestResnetResidual

lean_exe «test-spec-helpers» where
  root := `TestSpecHelpers

lean_exe «test-smoke-trainers» where
  root := `TestSmokeTrainers

lean_exe «test-codegen-ts» where
  root := `TestCodegenTrainStep

-- ════════════════════════════════════════════════════════════════
-- Bestiary: architecture-only NetSpec examples (print, no training)
-- ════════════════════════════════════════════════════════════════

lean_exe «bestiary-alphazero» where
  root := `Bestiary.AlphaZero

lean_exe «bestiary-mamba» where
  root := `Bestiary.Mamba

lean_exe «bestiary-swin» where
  root := `Bestiary.SwinT

lean_exe «bestiary-unet» where
  root := `Bestiary.UNet

lean_exe «bestiary-detr» where
  root := `Bestiary.DETR

lean_exe «bestiary-yolo» where
  root := `Bestiary.YOLO

lean_exe «bestiary-shufflenet» where
  root := `Bestiary.ShuffleNet

lean_exe «bestiary-evoformer» where
  root := `Bestiary.Evoformer

lean_exe «bestiary-muzero» where
  root := `Bestiary.MuZero

lean_exe «bestiary-mobilevit» where
  root := `Bestiary.MobileViT

lean_exe «bestiary-convnext» where
  root := `Bestiary.ConvNeXt

lean_exe «bestiary-wavenet» where
  root := `Bestiary.WaveNet

lean_exe «bestiary-nerf» where
  root := `Bestiary.NeRF

lean_exe «bestiary-clip» where
  root := `Bestiary.CLIP

lean_exe «bestiary-squeezenet» where
  root := `Bestiary.SqueezeNet

lean_exe «bestiary-lenet» where
  root := `Bestiary.LeNet

lean_exe «bestiary-inception» where
  root := `Bestiary.Inception

lean_exe «bestiary-xception» where
  root := `Bestiary.Xception

lean_exe «bestiary-alexnet» where
  root := `Bestiary.AlexNet

lean_exe «bestiary-bert» where
  root := `Bestiary.BERT

lean_exe «bestiary-shufflenetv2» where
  root := `Bestiary.ShuffleNetV2

lean_exe «bestiary-gpt» where
  root := `Bestiary.GPT

require checkdecls from git "https://github.com/PatrickMassot/checkdecls.git"