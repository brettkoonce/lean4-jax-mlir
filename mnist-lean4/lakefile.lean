import Lake
open Lake DSL System

package «mnist» where
  version := v!"0.1.0"
  buildType := .release

lean_exe «mnist-mlp» where
  root := `Main_working_1d_s4tf

-- Compile blascall.c → libblascall.a and link into mnist-cnn
target blascallLib pkg : FilePath := do
  let oFile  := pkg.buildDir / "blascall.o"
  let libFile := pkg.sharedLibDir / (nameToStaticLib "blascall")
  let srcJob ← inputTextFile (pkg.dir / "blascall.c")
  let leanInc := (← getLeanIncludeDir).toString
  let oJob ← buildO oFile srcJob
    #["-I", leanInc, "-I/usr/include/x86_64-linux-gnu"]
    #["-O2"]
    "cc"
    getLeanTrace
  buildStaticLib libFile #[oJob]

lean_exe «mnist-cnn» where
  root := `Main_working_2d_s4tf
  moreLinkObjs := #[blascallLib]
  moreLinkArgs := #["/lib/x86_64-linux-gnu/libopenblas.so"]

lean_exe «cifar-cnn» where
  root := `Main_cifar_v2
  moreLinkObjs := #[blascallLib]
  moreLinkArgs := #["/lib/x86_64-linux-gnu/libopenblas.so"]
