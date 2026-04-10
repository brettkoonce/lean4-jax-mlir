import Lake
open Lake DSL

package «lean4-jax» where
  version := v!"0.1.0"
  buildType := .release

lean_lib «LeanJax» where
  roots := #[`LeanJax]

-- Phase 2: Lean → JAX Python codegen + runner. Lives under jax/.
lean_lib «Jax» where
  roots := #[`Jax]
  srcDir := "jax"

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

lean_exe «mnist-mlp» where
  root := `MainMlp
  srcDir := "jax"

lean_exe «mnist-cnn» where
  root := `MainCnn
  srcDir := "jax"

lean_exe «cifar-cnn» where
  root := `MainCifar
  srcDir := "jax"

lean_exe «resnet34» where
  root := `MainResnet
  srcDir := "jax"

lean_exe «resnet34-train» where
  root := `MainResnetTrain
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «test-forward» where
  root := `TestForward
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «bench-resnet» where
  root := `BenchResnet
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «resnet50» where
  root := `MainResnet50
  srcDir := "jax"

lean_exe «resnet50-train» where
  root := `MainResnet50Train
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «mobilenet-v1» where
  root := `MainMobilenet
  srcDir := "jax"

lean_exe «mobilenet-v2» where
  root := `MainMobilenetV2
  srcDir := "jax"

lean_exe «mobilenet-v2-train» where
  root := `MainMobilenetV2Train
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «efficientnet-train» where
  root := `MainEfficientNetTrain
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «efficientnet-b0» where
  root := `MainEfficientNet
  srcDir := "jax"

lean_exe «efficientnet-v2-train» where
  root := `MainEfficientNetV2Train
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «mobilenet-v3» where
  root := `MainMobilenetV3
  srcDir := "jax"

lean_exe «mobilenet-v3-train» where
  root := `MainMobilenetV3Train
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «squeezenet» where
  root := `MainSqueezeNet
  srcDir := "jax"

lean_exe «vgg16bn» where
  root := `MainVgg
  srcDir := "jax"

lean_exe «vgg-train» where
  root := `MainVggTrain
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «vit-tiny» where
  root := `MainVit
  srcDir := "jax"

lean_exe «vit-tiny-train» where
  root := `MainVitTrain
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «efficientnet-v2s» where
  root := `MainEfficientNetV2
  srcDir := "jax"

lean_exe «mobilenet-v4» where
  root := `MainMobilenetV4
  srcDir := "jax"

lean_exe «mobilenet-v4-train» where
  root := `MainMobilenetV4Train
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «mnist-cnn-mlir» where
  root := `MainCnnMlir

lean_exe «mnist-mlp-mlir» where
  root := `MainMlpMlir
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «test-iree» where
  root := `TestIreeRuntime
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «test-train» where
  root := `TestTrainStep
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «cifar-bn-train» where
  root := `MainCifarCnnBnTrain
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «mnist-cnn-train» where
  root := `MainMnistCnnTrain
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «test-iree-load» where
  root := `TestIreeLoad
  moreLinkArgs := #["-L", "/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi", "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «test-resnet-fwd» where
  root := `TestResnetForward

lean_exe «test-resnet-residual» where
  root := `TestResnetResidual

lean_exe «test-codegen-ts» where
  root := `TestCodegenTrainStep

lean_exe «test-f32» where
  root := `TestF32
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «mnist-mlp-train-f32» where
  root := `MainMlpTrainF32
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «cifar-cnn-train-f32» where
  root := `MainCifarTrainF32
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «cifar-cnn-train» where
  root := `MainCifarTrain
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]

lean_exe «mnist-mlp-train» where
  root := `MainMlpTrain
  moreLinkArgs := #[
    "-L/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-liree_ffi",
    "-Wl,-rpath,/home/skoonce/lean/claude_max/lean4-jax/ffi",
    "-Wl,--allow-shlib-undefined"]
