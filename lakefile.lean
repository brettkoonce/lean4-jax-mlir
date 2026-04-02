import Lake
open Lake DSL

package «lean4-jax» where
  version := v!"0.1.0"
  buildType := .release

lean_lib «LeanJax» where
  roots := #[`LeanJax]

lean_exe «mnist-mlp» where
  root := `MainMlp

lean_exe «mnist-cnn» where
  root := `MainCnn

lean_exe «cifar-cnn» where
  root := `MainCifar

lean_exe «resnet34» where
  root := `MainResnet

lean_exe «resnet50» where
  root := `MainResnet50

lean_exe «mobilenet-v1» where
  root := `MainMobilenet
