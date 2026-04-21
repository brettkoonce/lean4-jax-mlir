import Lake
open Lake DSL

package «jax» where
  version := v!"0.1.0"
  buildType := .release

-- Depend on the parent package for LeanMlir.Types and LeanMlir.Spec.
require «lean4-jax» from ".."

lean_lib «Jax» where
  roots := #[`Jax]

-- Phase 2 JAX codegen runners — one per architecture.
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

lean_exe «mobilenet-v2» where
  root := `MainMobilenetV2

lean_exe «mobilenet-v3» where
  root := `MainMobilenetV3

lean_exe «mobilenet-v4» where
  root := `MainMobilenetV4

lean_exe «efficientnet-b0» where
  root := `MainEfficientNet

lean_exe «efficientnet-v2s» where
  root := `MainEfficientNetV2

lean_exe «squeezenet» where
  root := `MainSqueezeNet

lean_exe «vgg16bn» where
  root := `MainVgg

lean_exe «vit-tiny» where
  root := `MainVit

-- VJP oracle — one binary per axiom under test. See tests/vjp_oracle/.
lean_exe «vjp-oracle-dense» where
  root := `MainVjpOracleDense

lean_exe «vjp-oracle-dense-relu» where
  root := `MainVjpOracleDenseRelu

lean_exe «vjp-oracle-conv» where
  root := `MainVjpOracleConv

lean_exe «vjp-oracle-convbn» where
  root := `MainVjpOracleConvBn

lean_exe «vjp-oracle-conv-pool» where
  root := `MainVjpOracleConvPool
