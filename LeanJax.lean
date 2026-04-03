/-!
# Lean 4 → JAX: Neural network specification and code generation.

Declare architecture and training config as Lean data structures.
Generate idiomatic JAX Python and run training.
-/

-- ===========================================================================
-- § 1  ML Specification Types
-- ===========================================================================

inductive Activation where
  | relu
  | relu6
  | identity
deriving Repr, BEq

inductive Padding where
  | same
  | valid
deriving Repr, BEq

inductive Layer where
  | conv2d  (ic oc kSize : Nat) (pad : Padding) (act : Activation)
  | convBn  (ic oc kSize stride : Nat) (pad : Padding)
  | maxPool (size stride : Nat)
  | globalAvgPool
  | flatten
  | dense   (fanIn fanOut : Nat) (act : Activation)
  | residualBlock (ic oc nBlocks firstStride : Nat)
  | bottleneckBlock (ic oc nBlocks firstStride : Nat)
  | separableConv (ic oc stride : Nat)
  | invertedResidual (ic oc expand stride nBlocks : Nat)
  | mbConv (ic oc expand kSize stride nBlocks : Nat) (useSE : Bool)
  | mbConvV3 (ic oc expandCh kSize stride : Nat) (useSE useHSwish : Bool)
  | fireModule (ic squeeze expand1x1 expand3x3 : Nat)
deriving Repr

structure NetSpec where
  name   : String
  layers : List Layer
  imageH : Nat := 28
  imageW : Nat := 28
deriving Repr

structure TrainConfig where
  learningRate : Float
  batchSize    : Nat
  epochs       : Nat
  seed         : Nat := 314159
  momentum     : Float := 0.0
  useAdam      : Bool := false
  weightDecay  : Float := 0.0
  cosineDecay  : Bool := false
  warmupEpochs : Nat := 0
deriving Repr

inductive DatasetKind where
  | mnist
  | cifar10
  | imagenette
deriving Repr, BEq

-- ===========================================================================
-- § 2  Spec Helpers
-- ===========================================================================

def Layer.nParams : Layer → Nat
  | .conv2d ic oc k _ _     => oc * ic * k * k + oc
  | .convBn ic oc k _ _     => oc * ic * k * k + 2 * oc
  | .dense fi fo _           => fi * fo + fo
  | .residualBlock ic oc n fs =>
      let needsProj := !(ic == oc && fs == 1)
      let idBlock := 2 * (oc * oc * 3 * 3 + 2 * oc)
      let firstBlock := if needsProj
        then (oc * ic * 3 * 3 + 2 * oc) + (oc * oc * 3 * 3 + 2 * oc) + (oc * ic * 1 * 1 + 2 * oc)
        else idBlock
      firstBlock + (n - 1) * idBlock
  | .bottleneckBlock ic oc n fs =>
      let mid := oc / 4
      let needsProj := !(ic == oc && fs == 1)
      let idBlock := (mid * oc * 1 + 2 * mid) + (mid * mid * 9 + 2 * mid) + (oc * mid * 1 + 2 * oc)
      let firstBlock := if needsProj
        then (mid * ic * 1 + 2 * mid) + (mid * mid * 9 + 2 * mid) + (oc * mid * 1 + 2 * oc) +
             (oc * ic * 1 + 2 * oc)
        else idBlock
      firstBlock + (n - 1) * idBlock
  | .separableConv ic oc _ =>
      -- depthwise 3x3: ic*9 weights + 2*ic (BN)
      -- pointwise 1x1: oc*ic weights + 2*oc (BN)
      (ic * 9 + 2 * ic) + (oc * ic + 2 * oc)
  | .invertedResidual ic oc expand stride n =>
      let mid := ic * expand
      let expandP := if expand == 1 then 0 else (mid * ic + 2 * mid)  -- 1x1 expand + BN
      let dwP := mid * 9 + 2 * mid                                     -- depthwise 3x3 + BN
      let projP := oc * mid + 2 * oc                                   -- 1x1 project + BN
      let firstBlock := expandP + dwP + projP
      -- Subsequent blocks: ic becomes oc, stride=1, expand stays
      let midR := oc * expand
      let expandR := if expand == 1 then 0 else (midR * oc + 2 * midR)
      let dwR := midR * 9 + 2 * midR
      let projR := oc * midR + 2 * oc
      let restBlock := expandR + dwR + projR
      firstBlock + (n - 1) * restBlock
  | .mbConv ic oc expand k stride n useSE =>
      let mid := ic * expand
      let expandP := if expand == 1 then 0 else (mid * ic + 2 * mid)
      let dwP := mid * k * k + 2 * mid
      let seMid := Nat.max 1 (mid / 4)
      let seP := if useSE then (seMid * mid + seMid) + (mid * seMid + mid) else 0
      let projP := oc * mid + 2 * oc
      let firstBlock := expandP + dwP + seP + projP
      let midR := oc * expand
      let expandR := if expand == 1 then 0 else (midR * oc + 2 * midR)
      let dwR := midR * k * k + 2 * midR
      let seMidR := Nat.max 1 (midR / 4)
      let seR := if useSE then (seMidR * midR + seMidR) + (midR * seMidR + midR) else 0
      let projR := oc * midR + 2 * oc
      let restBlock := expandR + dwR + seR + projR
      firstBlock + (n - 1) * restBlock
  | .mbConvV3 ic oc expandCh k _ useSE _ =>
      let expandP := if expandCh == ic then 0 else (expandCh * ic + 2 * expandCh)
      let dwP := expandCh * k * k + 2 * expandCh
      let seMid := Nat.max 1 (expandCh / 4)
      let seP := if useSE then (seMid * expandCh + seMid) + (expandCh * seMid + expandCh) else 0
      let projP := oc * expandCh + 2 * oc
      expandP + dwP + seP + projP
  | .fireModule ic sq e1 e3 =>
      (sq * ic + 2 * sq) + (e1 * sq + 2 * e1) + (e3 * sq * 9 + 2 * e3)
  | _                        => 0

def NetSpec.totalParams (s : NetSpec) : Nat :=
  s.layers.foldl (fun acc l => acc + l.nParams) 0

def NetSpec.hasConv (s : NetSpec) : Bool :=
  s.layers.any fun | .conv2d .. => true | .convBn .. => true | .residualBlock .. => true | .bottleneckBlock .. => true | .separableConv .. => true | .invertedResidual .. => true | .mbConv .. => true | .mbConvV3 .. => true | .fireModule .. => true | _ => false

def NetSpec.hasPool (s : NetSpec) : Bool :=
  s.layers.any fun | .maxPool .. => true | _ => false

def NetSpec.hasBn (s : NetSpec) : Bool :=
  s.layers.any fun | .convBn .. => true | .residualBlock .. => true | .bottleneckBlock .. => true | .separableConv .. => true | .invertedResidual .. => true | .mbConv .. => true | .mbConvV3 .. => true | .fireModule .. => true | _ => false

def NetSpec.hasResidual (s : NetSpec) : Bool :=
  s.layers.any fun | .residualBlock .. => true | .bottleneckBlock .. => true | _ => false

def NetSpec.hasBottleneck (s : NetSpec) : Bool :=
  s.layers.any fun | .bottleneckBlock .. => true | _ => false

def NetSpec.hasSeparable (s : NetSpec) : Bool :=
  s.layers.any fun | .separableConv .. => true | .invertedResidual .. => true | .mbConv .. => true | .mbConvV3 .. => true | _ => false

def NetSpec.hasInvertedResidual (s : NetSpec) : Bool :=
  s.layers.any fun | .invertedResidual .. => true | _ => false

def NetSpec.hasMbConv (s : NetSpec) : Bool :=
  s.layers.any fun | .mbConv .. => true | .mbConvV3 .. => true | _ => false

def NetSpec.hasGlobalAvgPool (s : NetSpec) : Bool :=
  s.layers.any fun | .globalAvgPool => true | _ => false

def NetSpec.numClasses (s : NetSpec) : Nat :=
  match s.layers.getLast? with
  | some (.dense _ fo _) => fo
  | _                     => 0

def NetSpec.archStr (s : NetSpec) : String :=
  " → ".intercalate (s.layers.map fun l =>
    match l with
    | .conv2d ic oc k _ _       => s!"Conv({ic}→{oc},{k}x{k})"
    | .convBn ic oc k st _      => s!"Conv({ic}→{oc},{k}x{k}/{st})+BN"
    | .maxPool sz _              => s!"Pool({sz}x{sz})"
    | .globalAvgPool             => "GAP"
    | .flatten                   => "Flatten"
    | .dense fi fo act           =>
      let a := match act with | .relu => ",ReLU" | .relu6 => ",ReLU6" | .identity => ""
      s!"{fi}→{fo}{a}"
    | .residualBlock ic oc n fs   => s!"Res{n}({ic}→{oc},s{fs})"
    | .bottleneckBlock ic oc n fs => s!"BN{n}({ic}→{oc},s{fs})"
    | .separableConv ic oc st        => s!"Sep({ic}→{oc},s{st})"
    | .invertedResidual ic oc e s n     => s!"IR{n}({ic}→{oc},e{e},s{s})"
    | .mbConv ic oc e k s n useSE      => s!"MB{n}({ic}→{oc},e{e},k{k},s{s}" ++ (if useSE then ",SE" else "") ++ ")"
    | .mbConvV3 ic oc exp k s useSE hs => s!"V3({ic}→{oc},{exp},k{k},s{s}" ++
        (if useSE then ",SE" else "") ++ (if hs then ",HS" else ",RE") ++ ")"
    | .fireModule ic sq e1 e3 => s!"Fire({ic}→{e1 + e3},sq{sq})")

-- ===========================================================================
-- § 3  JAX Code Generator
-- ===========================================================================

namespace JaxCodegen

private def emitHeader (spec : NetSpec) : String :=
  "\"\"\"" ++ spec.name ++ " — Generated by Lean 4 → JAX\n" ++
  "Architecture: " ++ spec.archStr ++ "\n" ++
  "Parameters: " ++ toString spec.totalParams ++ "\n" ++
  "Matching: Convolutional Neural Networks with Swift for TensorFlow\n" ++
  "\"\"\"\n"

private def emitImports : String :=
  "import jax\n" ++
  "import jax.numpy as jnp\n" ++
  "from jax import random, jit, value_and_grad\n" ++
  "from jax.sharding import Mesh, NamedSharding, PartitionSpec as P\n" ++
  "import numpy as np\n" ++
  "import struct, os, time\n\n"

private def emitDataLoading (ds : DatasetKind) : String :=
  match ds with
  | .mnist =>
    "# ═══════════════════════════════════════════════════════════════════════\n" ++
    "#  MNIST data loading (raw IDX format)\n" ++
    "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
    "def load_mnist_images(path):\n" ++
    "    with open(path, \"rb\") as f:\n" ++
    "        magic, n, rows, cols = struct.unpack(\">4I\", f.read(16))\n" ++
    "        assert magic == 2051\n" ++
    "        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0\n\n" ++
    "def load_mnist_labels(path):\n" ++
    "    with open(path, \"rb\") as f:\n" ++
    "        magic, n = struct.unpack(\">2I\", f.read(8))\n" ++
    "        assert magic == 2049\n" ++
    "        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)\n\n"
  | .cifar10 =>
    "# ═══════════════════════════════════════════════════════════════════════\n" ++
    "#  CIFAR-10 data loading (binary batch format)\n" ++
    "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
    "def load_cifar10_batch(path):\n" ++
    "    with open(path, \"rb\") as f:\n" ++
    "        data = np.frombuffer(f.read(), dtype=np.uint8)\n" ++
    "    n = len(data) // 3073\n" ++
    "    data = data.reshape(n, 3073)\n" ++
    "    labels = data[:, 0].astype(np.int32)\n" ++
    "    images = data[:, 1:].astype(np.float32) / 255.0\n" ++
    "    return images, labels\n\n" ++
    "def load_cifar10(data_dir):\n" ++
    "    train_imgs, train_lbls = [], []\n" ++
    "    for i in range(1, 6):\n" ++
    "        imgs, lbls = load_cifar10_batch(os.path.join(data_dir, \"data_batch_\" + str(i) + \".bin\"))\n" ++
    "        train_imgs.append(imgs)\n" ++
    "        train_lbls.append(lbls)\n" ++
    "    test_imgs, test_lbls = load_cifar10_batch(os.path.join(data_dir, \"test_batch.bin\"))\n" ++
    "    return np.concatenate(train_imgs), np.concatenate(train_lbls), test_imgs, test_lbls\n\n"
  | .imagenette =>
    "# ═══════════════════════════════════════════════════════════════════════\n" ++
    "#  Imagenette data loading (binary format: 224×224×3, CHW)\n" ++
    "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
    "def load_imagenette(path):\n" ++
    "    with open(path, \"rb\") as f:\n" ++
    "        count = struct.unpack(\"<I\", f.read(4))[0]\n" ++
    "        data = np.frombuffer(f.read(), dtype=np.uint8)\n" ++
    "    record_size = 1 + 224 * 224 * 3\n" ++
    "    data = data.reshape(count, record_size)\n" ++
    "    labels = data[:, 0].astype(np.int32)\n" ++
    "    images = data[:, 1:].astype(np.float32) / 255.0\n" ++
    "    # Normalize with ImageNet statistics (data is CHW)\n" ++
    "    images = images.reshape(count, 3, 224, 224)\n" ++
    "    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)\n" ++
    "    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)\n" ++
    "    images = ((images - mean) / std).reshape(count, -1)\n" ++
    "    return images, labels\n\n"

private def emitHelpers (spec : NetSpec) : String := Id.run do
  let mut code := ""
  if spec.hasConv then
    code := code ++
      "# ═══════════════════════════════════════════════════════════════════════\n" ++
      "#  Conv / Pool helpers\n" ++
      "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
      "def conv2d(x, w, b, padding='SAME', stride=(1,1)):\n" ++
      "    x = jax.lax.conv_general_dilated(x, w, stride, padding,\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "    return x + b.reshape(1, -1, 1, 1)\n\n"
  if spec.hasPool then
    code := code ++
      "def max_pool2d(x, size=2, stride=2):\n" ++
      "    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,\n" ++
      "             (1, 1, size, size), (1, 1, stride, stride), 'SAME')\n\n"
  if spec.hasBn then
    code := code ++
      "def conv_bn(x, w, gamma, beta, stride=(1,1), padding='SAME'):\n" ++
      "    x = jax.lax.conv_general_dilated(x, w, stride, padding,\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "    # Instance normalization (per-sample, spatial stats)\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    return x * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)\n\n"
  if spec.hasResidual then
    code := code ++
      "def basic_block(params, x, idx):\n" ++
      "    out = conv_bn(x, params[idx][0], params[idx][1], params[idx][2])\n" ++
      "    out = jax.nn.relu(out)\n" ++
      "    out = conv_bn(out, params[idx+1][0], params[idx+1][1], params[idx+1][2])\n" ++
      "    return jax.nn.relu(out + x)\n\n" ++
      "def basic_block_down(params, x, idx, stride):\n" ++
      "    out = conv_bn(x, params[idx][0], params[idx][1], params[idx][2], stride=(stride,stride))\n" ++
      "    out = jax.nn.relu(out)\n" ++
      "    out = conv_bn(out, params[idx+1][0], params[idx+1][1], params[idx+1][2])\n" ++
      "    shortcut = conv_bn(x, params[idx+2][0], params[idx+2][1], params[idx+2][2], stride=(stride,stride))\n" ++
      "    return jax.nn.relu(out + shortcut)\n\n"
  if spec.hasBottleneck then
    code := code ++
      "def bottleneck_block(params, x, idx):\n" ++
      "    out = conv_bn(x, params[idx][0], params[idx][1], params[idx][2])\n" ++
      "    out = jax.nn.relu(out)\n" ++
      "    out = conv_bn(out, params[idx+1][0], params[idx+1][1], params[idx+1][2])\n" ++
      "    out = jax.nn.relu(out)\n" ++
      "    out = conv_bn(out, params[idx+2][0], params[idx+2][1], params[idx+2][2])\n" ++
      "    return jax.nn.relu(out + x)\n\n" ++
      "def bottleneck_block_down(params, x, idx, stride):\n" ++
      "    out = conv_bn(x, params[idx][0], params[idx][1], params[idx][2])\n" ++
      "    out = jax.nn.relu(out)\n" ++
      "    out = conv_bn(out, params[idx+1][0], params[idx+1][1], params[idx+1][2], stride=(stride,stride))\n" ++
      "    out = jax.nn.relu(out)\n" ++
      "    out = conv_bn(out, params[idx+2][0], params[idx+2][1], params[idx+2][2])\n" ++
      "    shortcut = conv_bn(x, params[idx+3][0], params[idx+3][1], params[idx+3][2], stride=(stride,stride))\n" ++
      "    return jax.nn.relu(out + shortcut)\n\n"
  if spec.hasSeparable then
    code := code ++
      "def depthwise_conv(x, w, stride=(1,1), padding='SAME'):\n" ++
      "    return jax.lax.conv_general_dilated(x, w, stride, padding,\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),\n" ++
      "          feature_group_count=x.shape[1])\n\n" ++
      "def sep_conv(x, dw, dw_g, dw_b, pw, pw_g, pw_b, stride=(1,1)):\n" ++
      "    \"\"\"Depthwise 3x3 + BN + ReLU6, then pointwise 1x1 + BN + ReLU6.\"\"\"\n" ++
      "    x = depthwise_conv(x, dw, stride=stride)\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * dw_g.reshape(1, -1, 1, 1) + dw_b.reshape(1, -1, 1, 1)\n" ++
      "    x = jnp.minimum(jax.nn.relu(x), 6.0)\n" ++
      "    x = jax.lax.conv_general_dilated(x, pw, (1,1), 'SAME',\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * pw_g.reshape(1, -1, 1, 1) + pw_b.reshape(1, -1, 1, 1)\n" ++
      "    return jnp.minimum(jax.nn.relu(x), 6.0)\n\n"
  if spec.hasInvertedResidual then
    code := code ++
      "def invres_block(params, x, idx, stride, expand, use_skip):\n" ++
      "    \"\"\"Inverted residual: expand 1x1 → depthwise 3x3 → project 1x1, + skip.\"\"\"\n" ++
      "    residual = x\n" ++
      "    i = idx\n" ++
      "    if expand > 1:\n" ++
      "        # Expand\n" ++
      "        x = jax.lax.conv_general_dilated(x, params[i][0], (1,1), 'SAME',\n" ++
      "              dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "        mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "        var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "        x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "        x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "        x = jnp.minimum(jax.nn.relu(x), 6.0)\n" ++
      "        i += 1\n" ++
      "    # Depthwise\n" ++
      "    x = depthwise_conv(x, params[i][0], stride=(stride,stride))\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "    x = jnp.minimum(jax.nn.relu(x), 6.0)\n" ++
      "    i += 1\n" ++
      "    # Project (linear, no activation)\n" ++
      "    x = jax.lax.conv_general_dilated(x, params[i][0], (1,1), 'SAME',\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "    if use_skip:\n" ++
      "        x = x + residual\n" ++
      "    return x\n\n"
  if spec.hasMbConv || spec.layers.any fun | .mbConvV3 .. => true | _ => false then
    code := code ++
      "def swish(x):\n" ++
      "    return x * jax.nn.sigmoid(x)\n\n" ++
      "def hard_swish(x):\n" ++
      "    return x * jnp.minimum(jax.nn.relu(x + 3.0), 6.0) / 6.0\n\n" ++
      "def hard_sigmoid(x):\n" ++
      "    return jnp.minimum(jax.nn.relu(x + 3.0), 6.0) / 6.0\n\n"
  if spec.hasMbConv then
    code := code ++
      "def mbconv_block(params, x, idx, stride, expand, ksize, use_se):\n" ++
      "    \"\"\"MBConv: expand → depthwise → SE → project, with Swish.\"\"\"\n" ++
      "    residual = x\n" ++
      "    i = idx\n" ++
      "    if expand > 1:\n" ++
      "        x = jax.lax.conv_general_dilated(x, params[i][0], (1,1), 'SAME',\n" ++
      "              dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "        mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "        var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "        x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "        x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "        x = swish(x)\n" ++
      "        i += 1\n" ++
      "    # Depthwise conv\n" ++
      "    pad = ((ksize - 1) // 2, (ksize - 1) // 2)\n" ++
      "    x = jax.lax.conv_general_dilated(x, params[i][0], (stride,stride), (pad,pad),\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),\n" ++
      "          feature_group_count=x.shape[1])\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "    x = swish(x)\n" ++
      "    i += 1\n" ++
      "    # Squeeze-and-Excitation\n" ++
      "    if use_se:\n" ++
      "        se = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "        se = jax.lax.conv_general_dilated(se, params[i][0], (1,1), 'SAME',\n" ++
      "              dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "        se = se + params[i][1].reshape(1, -1, 1, 1)\n" ++
      "        se = swish(se)\n" ++
      "        i += 1\n" ++
      "        se = jax.lax.conv_general_dilated(se, params[i][0], (1,1), 'SAME',\n" ++
      "              dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "        se = se + params[i][1].reshape(1, -1, 1, 1)\n" ++
      "        se = jax.nn.sigmoid(se)\n" ++
      "        x = x * se\n" ++
      "        i += 1\n" ++
      "    # Project (linear)\n" ++
      "    x = jax.lax.conv_general_dilated(x, params[i][0], (1,1), 'SAME',\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "    if residual.shape == x.shape and stride == 1:\n" ++
      "        x = x + residual\n" ++
      "    return x\n\n"
  if spec.layers.any fun | .mbConvV3 .. => true | _ => false then
    code := code ++
      "def mbconv_v3_block(params, x, idx, stride, expand_ch, ksize, use_se, use_hs):\n" ++
      "    \"\"\"MBConv v3: hard-swish/relu, hard-sigmoid SE.\"\"\"\n" ++
      "    act = hard_swish if use_hs else jax.nn.relu\n" ++
      "    residual = x\n" ++
      "    i = idx\n" ++
      "    ic = x.shape[1]\n" ++
      "    if expand_ch != ic:\n" ++
      "        x = jax.lax.conv_general_dilated(x, params[i][0], (1,1), 'SAME',\n" ++
      "              dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "        mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "        var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "        x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "        x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "        x = act(x)\n" ++
      "        i += 1\n" ++
      "    # Depthwise\n" ++
      "    pad = ((ksize - 1) // 2, (ksize - 1) // 2)\n" ++
      "    x = jax.lax.conv_general_dilated(x, params[i][0], (stride,stride), (pad,pad),\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'),\n" ++
      "          feature_group_count=x.shape[1])\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "    x = act(x)\n" ++
      "    i += 1\n" ++
      "    if use_se:\n" ++
      "        se = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "        se = jax.lax.conv_general_dilated(se, params[i][0], (1,1), 'SAME',\n" ++
      "              dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "        se = se + params[i][1].reshape(1, -1, 1, 1)\n" ++
      "        se = jax.nn.relu(se)\n" ++
      "        i += 1\n" ++
      "        se = jax.lax.conv_general_dilated(se, params[i][0], (1,1), 'SAME',\n" ++
      "              dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "        se = se + params[i][1].reshape(1, -1, 1, 1)\n" ++
      "        se = hard_sigmoid(se)\n" ++
      "        x = x * se\n" ++
      "        i += 1\n" ++
      "    # Project (linear)\n" ++
      "    x = jax.lax.conv_general_dilated(x, params[i][0], (1,1), 'SAME',\n" ++
      "          dimension_numbers=('NCHW', 'OIHW', 'NCHW'))\n" ++
      "    mean = jnp.mean(x, axis=(2, 3), keepdims=True)\n" ++
      "    var = jnp.var(x, axis=(2, 3), keepdims=True)\n" ++
      "    x = (x - mean) / jnp.sqrt(var + 1e-5)\n" ++
      "    x = x * params[i][1].reshape(1, -1, 1, 1) + params[i][2].reshape(1, -1, 1, 1)\n" ++
      "    if residual.shape == x.shape and stride == 1:\n" ++
      "        x = x + residual\n" ++
      "    return x\n\n"
  if spec.layers.any fun | .fireModule .. => true | _ => false then
    code := code ++
      "def fire_module(params, x, idx):\n" ++
      "    \"\"\"Fire: squeeze 1x1 → expand (1x1 || 3x3) → concat.\"\"\"\n" ++
      "    sq = conv_bn(x, params[idx][0], params[idx][1], params[idx][2])\n" ++
      "    sq = jax.nn.relu(sq)\n" ++
      "    e1 = conv_bn(sq, params[idx+1][0], params[idx+1][1], params[idx+1][2])\n" ++
      "    e1 = jax.nn.relu(e1)\n" ++
      "    e3 = conv_bn(sq, params[idx+2][0], params[idx+2][1], params[idx+2][2])\n" ++
      "    e3 = jax.nn.relu(e3)\n" ++
      "    return jnp.concatenate([e1, e3], axis=1)\n\n"
  if spec.hasGlobalAvgPool then
    code := code ++
      "def global_avg_pool(x):\n" ++
      "    return jnp.mean(x, axis=(2, 3))\n\n"
  code

-- Helper: emit init code for one conv+BN param group (w, gamma, beta)
private def emitConvBnInit (comment : String) (ic oc k : Nat) (zeroGamma : Bool := false) : String :=
  let fanOut := oc * k * k
  let gammaInit := if zeroGamma then "jnp.zeros(" ++ toString oc ++ ")" else "jnp.ones(" ++ toString oc ++ ")"
  "    # " ++ comment ++ "\n" ++
  "    key, k_ = random.split(key)\n" ++
  "    scale = jnp.sqrt(6.0 / " ++ toString fanOut ++ ")\n" ++
  "    params.append((random.uniform(k_, (" ++ toString oc ++ ", " ++ toString ic ++ ", " ++
    toString k ++ ", " ++ toString k ++ "), minval=-scale, maxval=scale),\n" ++
  "                   " ++ gammaInit ++ ", jnp.zeros(" ++ toString oc ++ ")))\n"

private def emitInitParams (spec : NetSpec) : String := Id.run do
  let mut code :=
    "# ═══════════════════════════════════════════════════════════════════════\n" ++
    "#  Model (from Lean spec)\n" ++
    "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
    "def init_params(key):\n" ++
    "    \"\"\"Xavier/Kaiming uniform init.\"\"\"\n" ++
    "    params = []\n"
  for l in spec.layers do
    match l with
    | .conv2d ic oc k _ _ =>
      let fanIn := ic * k * k
      let fanOut := oc * k * k
      code := code ++
        "    # Conv2D " ++ toString ic ++ "→" ++ toString oc ++ ", " ++ toString k ++ "x" ++ toString k ++ "\n" ++
        "    key, k_ = random.split(key)\n" ++
        "    scale = jnp.sqrt(6.0 / " ++ toString (fanIn + fanOut) ++ ")\n" ++
        "    params.append((random.uniform(k_, (" ++ toString oc ++ ", " ++ toString ic ++ ", " ++
          toString k ++ ", " ++ toString k ++ "), minval=-scale, maxval=scale), jnp.zeros(" ++ toString oc ++ ")))\n"
    | .convBn ic oc k _ _ =>
      code := code ++ emitConvBnInit s!"ConvBN {ic}→{oc}, {k}x{k}" ic oc k
    | .dense fi fo _ =>
      code := code ++
        "    # Dense " ++ toString fi ++ "→" ++ toString fo ++ "\n" ++
        "    key, k_ = random.split(key)\n" ++
        "    scale = jnp.sqrt(6.0 / " ++ toString (fi + fo) ++ ")\n" ++
        "    params.append((random.uniform(k_, (" ++ toString fo ++ ", " ++ toString fi ++
          "), minval=-scale, maxval=scale), jnp.zeros(" ++ toString fo ++ ")))\n"
    | .residualBlock ic oc n fs =>
      let needsProj := !(ic == oc && fs == 1)
      if needsProj then
        code := code ++ emitConvBnInit s!"ResBlock ↓ conv1 {ic}→{oc}" ic oc 3
        code := code ++ emitConvBnInit s!"ResBlock ↓ conv2 {oc}→{oc}" oc oc 3
        code := code ++ emitConvBnInit s!"ResBlock ↓ proj {ic}→{oc}" ic oc 1
      else
        code := code ++ emitConvBnInit s!"ResBlock conv1 {oc}→{oc}" oc oc 3
        code := code ++ emitConvBnInit s!"ResBlock conv2 {oc}→{oc}" oc oc 3
      for _ in List.range (n - 1) do
        code := code ++ emitConvBnInit s!"ResBlock conv1 {oc}→{oc}" oc oc 3
        code := code ++ emitConvBnInit s!"ResBlock conv2 {oc}→{oc}" oc oc 3
    | .bottleneckBlock ic oc n fs =>
      let mid := oc / 4
      let needsProj := !(ic == oc && fs == 1)
      if needsProj then
        code := code ++ emitConvBnInit s!"Bottleneck ↓ 1x1 {ic}→{mid}" ic mid 1
        code := code ++ emitConvBnInit s!"Bottleneck ↓ 3x3 {mid}→{mid}" mid mid 3
        code := code ++ emitConvBnInit s!"Bottleneck ↓ 1x1 {mid}→{oc}" mid oc 1 (zeroGamma := true)
        code := code ++ emitConvBnInit s!"Bottleneck ↓ proj {ic}→{oc}" ic oc 1
      else
        code := code ++ emitConvBnInit s!"Bottleneck 1x1 {oc}→{mid}" oc mid 1
        code := code ++ emitConvBnInit s!"Bottleneck 3x3 {mid}→{mid}" mid mid 3
        code := code ++ emitConvBnInit s!"Bottleneck 1x1 {mid}→{oc}" mid oc 1 (zeroGamma := true)
      for _ in List.range (n - 1) do
        code := code ++ emitConvBnInit s!"Bottleneck 1x1 {oc}→{mid}" oc mid 1
        code := code ++ emitConvBnInit s!"Bottleneck 3x3 {mid}→{mid}" mid mid 3
        code := code ++ emitConvBnInit s!"Bottleneck 1x1 {mid}→{oc}" mid oc 1 (zeroGamma := true)
    | .separableConv ic oc _ =>
      -- Depthwise: weight shape (ic, 1, 3, 3), Kaiming init
      let dwFanOut := 9
      code := code ++
        "    # SepConv DW " ++ toString ic ++ "ch, 3x3\n" ++
        "    key, k_ = random.split(key)\n" ++
        "    scale = jnp.sqrt(6.0 / " ++ toString dwFanOut ++ ")\n" ++
        "    params.append((random.uniform(k_, (" ++ toString ic ++ ", 1, 3, 3), minval=-scale, maxval=scale),\n" ++
        "                   jnp.ones(" ++ toString ic ++ "), jnp.zeros(" ++ toString ic ++ ")))\n"
      -- Pointwise: reuse emitConvBnInit for 1x1 conv
      code := code ++ emitConvBnInit s!"SepConv PW {ic}→{oc}" ic oc 1
    | .invertedResidual ic oc expand stride n =>
      -- First block: ic→mid→oc
      let mid := ic * expand
      if expand != 1 then
        code := code ++ emitConvBnInit s!"IR expand {ic}→{mid}" ic mid 1
      -- Depthwise
      let dwFanOut := 9
      code := code ++
        "    # IR DW " ++ toString mid ++ "ch, 3x3\n" ++
        "    key, k_ = random.split(key)\n" ++
        "    scale = jnp.sqrt(6.0 / " ++ toString dwFanOut ++ ")\n" ++
        "    params.append((random.uniform(k_, (" ++ toString mid ++ ", 1, 3, 3), minval=-scale, maxval=scale),\n" ++
        "                   jnp.ones(" ++ toString mid ++ "), jnp.zeros(" ++ toString mid ++ ")))\n"
      code := code ++ emitConvBnInit s!"IR project {mid}→{oc}" mid oc 1
      -- Remaining blocks: oc→oc*expand→oc, stride=1
      for _ in List.range (n - 1) do
        let midR := oc * expand
        if expand != 1 then
          code := code ++ emitConvBnInit s!"IR expand {oc}→{midR}" oc midR 1
        code := code ++
          "    # IR DW " ++ toString midR ++ "ch, 3x3\n" ++
          "    key, k_ = random.split(key)\n" ++
          "    scale = jnp.sqrt(6.0 / " ++ toString dwFanOut ++ ")\n" ++
          "    params.append((random.uniform(k_, (" ++ toString midR ++ ", 1, 3, 3), minval=-scale, maxval=scale),\n" ++
          "                   jnp.ones(" ++ toString midR ++ "), jnp.zeros(" ++ toString midR ++ ")))\n"
        code := code ++ emitConvBnInit s!"IR project {midR}→{oc}" midR oc 1
    | .mbConv ic oc expand kSize _ n useSE =>
      -- Helper to emit one block's params
      let emitBlock (blockIc blockOc : Nat) : String := Id.run do
        let mid := blockIc * expand
        let mut code := ""
        -- Expand
        if expand != 1 then
          code := code ++ emitConvBnInit s!"MB expand {blockIc}→{mid}" blockIc mid 1
        -- Depthwise
        let dwFanOut := kSize * kSize
        code := code ++
          "    # MB DW " ++ toString mid ++ "ch, " ++ toString kSize ++ "x" ++ toString kSize ++ "\n" ++
          "    key, k_ = random.split(key)\n" ++
          "    scale = jnp.sqrt(6.0 / " ++ toString dwFanOut ++ ")\n" ++
          "    params.append((random.uniform(k_, (" ++ toString mid ++ ", 1, " ++
            toString kSize ++ ", " ++ toString kSize ++ "), minval=-scale, maxval=scale),\n" ++
          "                   jnp.ones(" ++ toString mid ++ "), jnp.zeros(" ++ toString mid ++ ")))\n"
        -- SE
        if useSE then
          let seMid := Nat.max 1 (mid / 4)
          code := code ++
            "    # SE down " ++ toString mid ++ "→" ++ toString seMid ++ "\n" ++
            "    key, k_ = random.split(key)\n" ++
            "    scale = jnp.sqrt(6.0 / " ++ toString seMid ++ ")\n" ++
            "    params.append((random.uniform(k_, (" ++ toString seMid ++ ", " ++ toString mid ++
              ", 1, 1), minval=-scale, maxval=scale), jnp.zeros(" ++ toString seMid ++ ")))\n" ++
            "    # SE up " ++ toString seMid ++ "→" ++ toString mid ++ "\n" ++
            "    key, k_ = random.split(key)\n" ++
            "    scale = jnp.sqrt(6.0 / " ++ toString mid ++ ")\n" ++
            "    params.append((random.uniform(k_, (" ++ toString mid ++ ", " ++ toString seMid ++
              ", 1, 1), minval=-scale, maxval=scale), jnp.zeros(" ++ toString mid ++ ")))\n"
        -- Project
        code := code ++ emitConvBnInit s!"MB project {mid}→{blockOc}" mid blockOc 1
        code
      -- First block
      code := code ++ emitBlock ic oc
      -- Remaining blocks
      for _ in List.range (n - 1) do
        code := code ++ emitBlock oc oc
    | .mbConvV3 ic oc expandCh kSize _ useSE _ =>
      -- Expand (if expandCh != ic)
      if expandCh != ic then
        code := code ++ emitConvBnInit s!"V3 expand {ic}→{expandCh}" ic expandCh 1
      -- Depthwise
      code := code ++
        "    # V3 DW " ++ toString expandCh ++ "ch, " ++ toString kSize ++ "x" ++ toString kSize ++ "\n" ++
        "    key, k_ = random.split(key)\n" ++
        "    scale = jnp.sqrt(6.0 / " ++ toString (kSize * kSize) ++ ")\n" ++
        "    params.append((random.uniform(k_, (" ++ toString expandCh ++ ", 1, " ++
          toString kSize ++ ", " ++ toString kSize ++ "), minval=-scale, maxval=scale),\n" ++
        "                   jnp.ones(" ++ toString expandCh ++ "), jnp.zeros(" ++ toString expandCh ++ ")))\n"
      -- SE
      if useSE then
        let seMid := Nat.max 1 (expandCh / 4)
        code := code ++
          "    # V3 SE down " ++ toString expandCh ++ "→" ++ toString seMid ++ "\n" ++
          "    key, k_ = random.split(key)\n" ++
          "    scale = jnp.sqrt(6.0 / " ++ toString seMid ++ ")\n" ++
          "    params.append((random.uniform(k_, (" ++ toString seMid ++ ", " ++ toString expandCh ++
            ", 1, 1), minval=-scale, maxval=scale), jnp.zeros(" ++ toString seMid ++ ")))\n" ++
          "    # V3 SE up " ++ toString seMid ++ "→" ++ toString expandCh ++ "\n" ++
          "    key, k_ = random.split(key)\n" ++
          "    scale = jnp.sqrt(6.0 / " ++ toString expandCh ++ ")\n" ++
          "    params.append((random.uniform(k_, (" ++ toString expandCh ++ ", " ++ toString seMid ++
            ", 1, 1), minval=-scale, maxval=scale), jnp.zeros(" ++ toString expandCh ++ ")))\n"
      -- Project
      code := code ++ emitConvBnInit s!"V3 project {expandCh}→{oc}" expandCh oc 1
    | .fireModule ic sq e1 e3 =>
      code := code ++ emitConvBnInit s!"Fire squeeze {ic}→{sq}" ic sq 1
      code := code ++ emitConvBnInit s!"Fire expand1x1 {sq}→{e1}" sq e1 1
      code := code ++ emitConvBnInit s!"Fire expand3x3 {sq}→{e3}" sq e3 3
    | _ => pure ()
  code := code ++ "    return params\n\n"
  code

private def emitForward (spec : NetSpec) : String := Id.run do
  let mut code := "def forward(params, x):\n"
  -- Reshape flat images to NCHW if first layer is conv
  match spec.layers.head? with
  | some (.conv2d ic _ _ _ _) =>
    code := code ++ "    x = x.reshape(-1, " ++ toString ic ++ ", " ++
      toString spec.imageH ++ ", " ++ toString spec.imageW ++ ")\n"
  | some (.convBn ic _ _ _ _) =>
    code := code ++ "    x = x.reshape(-1, " ++ toString ic ++ ", " ++
      toString spec.imageH ++ ", " ++ toString spec.imageW ++ ")\n"
  | _ => pure ()
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .conv2d _ _ _ pad act =>
      let padStr := match pad with | .same => "SAME" | .valid => "VALID"
      code := code ++ "    x = conv2d(x, params[" ++ toString pidx ++ "][0], params[" ++
        toString pidx ++ "][1], '" ++ padStr ++ "')\n"
      if act == .relu then code := code ++ "    x = jax.nn.relu(x)\n"
      if act == .relu6 then code := code ++ "    x = jnp.minimum(jax.nn.relu(x), 6.0)\n"
      pidx := pidx + 1
    | .convBn _ _ _ s pad =>
      let padStr := match pad with | .same => "SAME" | .valid => "VALID"
      let strideStr := if s == 1 then "" else ", stride=(" ++ toString s ++ "," ++ toString s ++ ")"
      code := code ++ "    x = conv_bn(x, params[" ++ toString pidx ++ "][0], params[" ++
        toString pidx ++ "][1], params[" ++ toString pidx ++ "][2]" ++
        strideStr ++ ", padding='" ++ padStr ++ "')\n"
      code := code ++ "    x = jax.nn.relu(x)\n"
      pidx := pidx + 1
    | .maxPool size stride =>
      code := code ++ "    x = max_pool2d(x, " ++ toString size ++ ", " ++ toString stride ++ ")\n"
    | .globalAvgPool =>
      code := code ++ "    x = global_avg_pool(x)\n"
    | .flatten =>
      code := code ++ "    x = x.reshape(x.shape[0], -1)\n"
    | .dense _ _ act =>
      code := code ++ "    x = x @ params[" ++ toString pidx ++ "][0].T + params[" ++
        toString pidx ++ "][1]\n"
      if act == .relu then code := code ++ "    x = jax.nn.relu(x)\n"
      if act == .relu6 then code := code ++ "    x = jnp.minimum(jax.nn.relu(x), 6.0)\n"
      pidx := pidx + 1
    | .residualBlock ic oc n fs =>
      let needsProj := !(ic == oc && fs == 1)
      if needsProj then
        code := code ++ "    x = basic_block_down(params, x, " ++ toString pidx ++ ", " ++ toString fs ++ ")\n"
        pidx := pidx + 3
      else
        code := code ++ "    x = basic_block(params, x, " ++ toString pidx ++ ")\n"
        pidx := pidx + 2
      for _ in List.range (n - 1) do
        code := code ++ "    x = basic_block(params, x, " ++ toString pidx ++ ")\n"
        pidx := pidx + 2
    | .bottleneckBlock ic oc n fs =>
      let needsProj := !(ic == oc && fs == 1)
      if needsProj then
        code := code ++ "    x = bottleneck_block_down(params, x, " ++ toString pidx ++ ", " ++ toString fs ++ ")\n"
        pidx := pidx + 4
      else
        code := code ++ "    x = bottleneck_block(params, x, " ++ toString pidx ++ ")\n"
        pidx := pidx + 3
      for _ in List.range (n - 1) do
        code := code ++ "    x = bottleneck_block(params, x, " ++ toString pidx ++ ")\n"
        pidx := pidx + 3
    | .separableConv _ _ s =>
      let strideStr := "(" ++ toString s ++ "," ++ toString s ++ ")"
      code := code ++ "    x = sep_conv(x, params[" ++ toString pidx ++ "][0], params[" ++
        toString pidx ++ "][1], params[" ++ toString pidx ++ "][2], params[" ++
        toString (pidx + 1) ++ "][0], params[" ++ toString (pidx + 1) ++ "][1], params[" ++
        toString (pidx + 1) ++ "][2], stride=" ++ strideStr ++ ")\n"
      pidx := pidx + 2
    | .mbConv ic oc expand kSize stride n useSE =>
      let nPerBlock (blockExpand : Nat) (se : Bool) :=
        (if blockExpand != 1 then 1 else 0) + 1 + (if se then 2 else 0) + 1
      let useSkipFirst := if ic == oc && stride == 1 then "True" else "False"
      let _ := useSkipFirst  -- used in string below
      code := code ++ "    x = mbconv_block(params, x, " ++ toString pidx ++ ", " ++
        toString stride ++ ", " ++ toString expand ++ ", " ++ toString kSize ++ ", " ++
        (if useSE then "True" else "False") ++ ")\n"
      pidx := pidx + nPerBlock expand useSE
      for _ in List.range (n - 1) do
        code := code ++ "    x = mbconv_block(params, x, " ++ toString pidx ++ ", 1, " ++
          toString expand ++ ", " ++ toString kSize ++ ", " ++
          (if useSE then "True" else "False") ++ ")\n"
        pidx := pidx + nPerBlock expand useSE
    | .invertedResidual ic oc expand stride n =>
      let nParamsPerBlock (hasExpand : Bool) := (if hasExpand then 1 else 0) + 1 + 1  -- expand? + dw + proj
      let hasExpand := expand != 1
      -- First block: may have stride, no skip (ic≠oc or stride≠1 typically)
      let useSkip := if ic == oc && stride == 1 then "True" else "False"
      code := code ++ "    x = invres_block(params, x, " ++ toString pidx ++ ", " ++
        toString stride ++ ", " ++ toString expand ++ ", " ++ useSkip ++ ")\n"
      pidx := pidx + nParamsPerBlock hasExpand
      -- Remaining blocks: stride=1, skip when oc==oc (always true)
      for _ in List.range (n - 1) do
        code := code ++ "    x = invres_block(params, x, " ++ toString pidx ++ ", 1, " ++
          toString expand ++ ", True)\n"
        pidx := pidx + nParamsPerBlock hasExpand
    | .mbConvV3 ic _ expandCh kSize stride useSE useHSwish =>
      let nP := (if expandCh != ic then 1 else 0) + 1 + (if useSE then 2 else 0) + 1
      code := code ++ "    x = mbconv_v3_block(params, x, " ++ toString pidx ++ ", " ++
        toString stride ++ ", " ++ toString expandCh ++ ", " ++ toString kSize ++ ", " ++
        (if useSE then "True" else "False") ++ ", " ++
        (if useHSwish then "True" else "False") ++ ")\n"
      pidx := pidx + nP
    | .fireModule _ _ _ _ =>
      code := code ++ "    x = fire_module(params, x, " ++ toString pidx ++ ")\n"
      pidx := pidx + 3
  code := code ++ "    return x\n\n"
  code

private def emitLossAndTraining (spec : NetSpec) (cfg : TrainConfig) : String :=
  let nClasses := spec.numClasses
  let lr := toString cfg.learningRate
  let hasMomentum := cfg.momentum > 0.0
  let hasCosine := cfg.cosineDecay
  "def loss_fn(params, x, y):\n" ++
  "    logits = forward(params, x)\n" ++
  "    log_probs = jax.nn.log_softmax(logits, axis=-1)\n" ++
  "    one_hot = jax.nn.one_hot(y, " ++ toString nClasses ++ ")\n" ++
  "    return -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))\n\n" ++
  let useAdam := cfg.useAdam
  let optName := if useAdam then "Adam" else "SGD" ++ (if hasMomentum then " + momentum" else "")
  "# ═══════════════════════════════════════════════════════════════════════\n" ++
  "#  Training  (" ++ optName ++
    (if hasCosine then " + cosine LR" else "") ++ ")\n" ++
  "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
  let hasWD := cfg.weightDecay > 0.0
  let wd := toString cfg.weightDecay
  "LR = " ++ lr ++ "\n" ++
  (if hasMomentum && !useAdam then "MOMENTUM = " ++ toString cfg.momentum ++ "\n" else "") ++
  (if hasWD then "WD = " ++ wd ++ "\n" else "") ++
  "\n" ++
  (if useAdam then
    "@jit\n" ++
    "def train_step(params, opt_state, x, y, lr):\n" ++
    "    loss, grads = value_and_grad(loss_fn)(params, x, y)\n" ++
    (if hasWD then "    grads = jax.tree.map(lambda g, p: g + WD * p, grads, params)\n" else "") ++
    "    m, v, t = opt_state\n" ++
    "    t = t + 1\n" ++
    "    m = jax.tree.map(lambda mi, g: 0.9 * mi + 0.1 * g, m, grads)\n" ++
    "    v = jax.tree.map(lambda vi, g: 0.999 * vi + 0.001 * g * g, v, grads)\n" ++
    "    mc = jax.tree.map(lambda mi: mi / (1 - 0.9 ** t), m)\n" ++
    "    vc = jax.tree.map(lambda vi: vi / (1 - 0.999 ** t), v)\n" ++
    "    params = jax.tree.map(lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + 1e-8), params, mc, vc)\n" ++
    "    return params, (m, v, t), loss\n\n"
  else if hasMomentum then
    "@jit\n" ++
    "def train_step(params, velocity, x, y, lr):\n" ++
    "    loss, grads = value_and_grad(loss_fn)(params, x, y)\n" ++
    (if hasWD then "    grads = jax.tree.map(lambda g, p: g + WD * p, grads, params)\n" else "") ++
    "    velocity = jax.tree.map(lambda v, g: MOMENTUM * v + g, velocity, grads)\n" ++
    "    params = jax.tree.map(lambda p, v: p - lr * v, params, velocity)\n" ++
    "    return params, velocity, loss\n\n"
  else
    "@jit\n" ++
    "def train_step(params, x, y, lr):\n" ++
    "    loss, grads = value_and_grad(loss_fn)(params, x, y)\n" ++
    (if hasWD then "    grads = jax.tree.map(lambda g, p: g + WD * p, grads, params)\n" else "") ++
    "    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)\n" ++
    "    return params, loss\n\n") ++
  "@jit\n" ++
  "def eval_batch(params, x, y):\n" ++
  "    logits = forward(params, x)\n" ++
  "    preds = jnp.argmax(logits, axis=-1)\n" ++
  "    log_probs = jax.nn.log_softmax(logits, axis=-1)\n" ++
  "    loss = -jnp.mean(jnp.sum(log_probs * jax.nn.one_hot(y, " ++ toString nClasses ++ "), axis=-1))\n" ++
  "    correct = jnp.sum(preds == y)\n" ++
  "    return correct, loss\n\n" ++
  "def evaluate(params, images, labels, batch_size=512):\n" ++
  "    batch_size = (batch_size // n_devices) * n_devices or n_devices\n" ++
  "    correct = 0\n" ++
  "    total_loss = 0.0\n" ++
  "    n_batches = 0\n" ++
  "    for i in range(0, len(images) - batch_size + 1, batch_size):\n" ++
  "        c, l = eval_batch(params,\n" ++
  "            jax.device_put(images[i:i+batch_size], data_sharding),\n" ++
  "            jax.device_put(labels[i:i+batch_size], data_sharding))\n" ++
  "        correct += int(c)\n" ++
  "        total_loss += float(l)\n" ++
  "        n_batches += 1\n" ++
  "    evaluated = n_batches * batch_size\n" ++
  "    return correct, evaluated, total_loss / max(n_batches, 1)\n\n"

private def emitDataLoadCalls (ds : DatasetKind) (dataDir : String) (spec : NetSpec) : String :=
  let imgDesc := toString spec.imageH ++ "x" ++ toString spec.imageW
  match ds with
  | .mnist =>
    "    data_dir = \"" ++ dataDir ++ "\"\n" ++
    "    print(\"Loading training set …\")\n" ++
    "    train_images = load_mnist_images(os.path.join(data_dir, \"train-images-idx3-ubyte\"))\n" ++
    "    train_labels = load_mnist_labels(os.path.join(data_dir, \"train-labels-idx1-ubyte\"))\n" ++
    "    print(\"  \" + str(len(train_images)) + \" images, " ++ imgDesc ++ "\")\n\n" ++
    "    print(\"Loading test set …\")\n" ++
    "    test_images = load_mnist_images(os.path.join(data_dir, \"t10k-images-idx3-ubyte\"))\n" ++
    "    test_labels = load_mnist_labels(os.path.join(data_dir, \"t10k-labels-idx1-ubyte\"))\n" ++
    "    print(\"  \" + str(len(test_images)) + \" images, " ++ imgDesc ++ "\")\n\n"
  | .cifar10 =>
    "    data_dir = \"" ++ dataDir ++ "\"\n" ++
    "    print(\"Loading CIFAR-10 …\")\n" ++
    "    train_images, train_labels, test_images, test_labels = load_cifar10(data_dir)\n" ++
    "    print(\"  \" + str(len(train_images)) + \" train, \" + str(len(test_images)) + \" test, " ++ imgDesc ++ "x3\")\n\n"
  | .imagenette =>
    "    data_dir = \"" ++ dataDir ++ "\"\n" ++
    "    print(\"Loading Imagenette …\")\n" ++
    "    train_images, train_labels = load_imagenette(os.path.join(data_dir, \"train.bin\"))\n" ++
    "    test_images, test_labels = load_imagenette(os.path.join(data_dir, \"val.bin\"))\n" ++
    "    print(\"  \" + str(len(train_images)) + \" train, \" + str(len(test_images)) + \" val, " ++ imgDesc ++ "x3\")\n\n"

private def emitMain (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind) (dataDir : String) : String :=
  let bs := toString cfg.batchSize
  let ep := toString cfg.epochs
  let lr := toString cfg.learningRate
  let nParams := toString spec.totalParams
  let seed := toString cfg.seed
  let hasMomentum := cfg.momentum > 0.0
  -- Pre-stage small datasets to GPU; keep large ones on CPU (batch transfer)
  let preStage := spec.imageH * spec.imageW ≤ 32 * 32
  "# ═══════════════════════════════════════════════════════════════════════\n" ++
  "#  Main\n" ++
  "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
  "if __name__ == \"__main__\":\n" ++
  "    print(\"" ++ spec.name ++ " · Lean 4 → JAX\")\n" ++
  "    print(\"  \" + \"" ++ spec.archStr ++ "\")\n" ++
  "    print()\n\n" ++
  emitDataLoadCalls ds dataDir spec ++
  "    BATCH_SIZE = (" ++ bs ++ " // n_devices) * n_devices or n_devices\n" ++
  "    EPOCHS = " ++ ep ++ "\n" ++
  "    print(\"lr=" ++ lr ++ "  batch_size=\" + str(BATCH_SIZE) + \" (\" + str(n_devices) + \" devices x \" + str(BATCH_SIZE // n_devices) + \")  epochs=\" + str(EPOCHS) + \"  params=" ++ nParams ++ "\")\n" ++
  "    print(\"backend=\" + str(jax.default_backend()) + \"  devices=\" + str(jax.devices()))\n" ++
  "    print(\"Starting training...\")\n\n" ++
  "    params = init_params(random.PRNGKey(" ++ seed ++ "))\n" ++
  "    params = jax.device_put(params, replicated_sharding)\n" ++
  "    rng = np.random.RandomState(42)\n" ++
  let useAdam := cfg.useAdam
  (if useAdam then
    "    opt_m = jax.tree.map(jnp.zeros_like, params)\n" ++
    "    opt_v = jax.tree.map(jnp.zeros_like, params)\n" ++
    "    opt_state = (opt_m, opt_v, jnp.float32(0))\n"
  else if hasMomentum then "    velocity = jax.tree.map(jnp.zeros_like, params)\n"
  else "") ++
  "\n" ++
  (if preStage then
  "    # Pre-stage all data as JAX arrays (one transfer, not per-batch)\n" ++
  "    train_images = jnp.array(train_images)\n" ++
  "    train_labels = jnp.array(train_labels)\n" ++
  "    test_images = jnp.array(test_images)\n" ++
  "    test_labels = jnp.array(test_labels)\n\n"
  else
  "    # Data stays on CPU; batches transferred to GPU on the fly\n\n") ++
  let warmup := toString cfg.warmupEpochs
  let hasCosine := cfg.cosineDecay
  "    t0 = time.time()\n" ++
  "    for epoch in range(EPOCHS):\n" ++
  (if hasCosine then
  "        # Cosine LR schedule" ++ (if cfg.warmupEpochs > 0 then " with warmup" else "") ++ "\n" ++
  (if cfg.warmupEpochs > 0 then
  "        if epoch < " ++ warmup ++ ":\n" ++
  "            lr = jnp.float32(LR * (epoch + 1) / " ++ warmup ++ ")\n" ++
  "        else:\n" ++
  "            lr = jnp.float32(LR * 0.5 * (1 + np.cos(np.pi * (epoch - " ++ warmup ++ ") / (EPOCHS - " ++ warmup ++ "))))\n"
  else
  "        lr = jnp.float32(LR * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS)))\n")
  else
  "        lr = jnp.float32(LR)\n") ++
  "        perm = rng.permutation(len(train_images))\n" ++
  "        shuf_images = train_images[perm]\n" ++
  "        shuf_labels = train_labels[perm]\n" ++
  let ic := match spec.layers.head? with
    | some (.conv2d ic ..) => ic | some (.convBn ic ..) => ic | _ => 3
  (if ¬preStage then  -- large images: random horizontal flip
  "        # Random horizontal flip (data augmentation)\n" ++
  "        flip = rng.random(len(shuf_images)) > 0.5\n" ++
  "        imgs4d = shuf_images.reshape(-1, " ++ toString ic ++ ", " ++
    toString spec.imageH ++ ", " ++ toString spec.imageW ++ ")\n" ++
  "        imgs4d[flip] = imgs4d[flip, :, :, ::-1]\n" ++
  "        shuf_images = imgs4d.reshape(len(shuf_images), -1)\n"
  else "") ++
  "        epoch_loss = 0.0\n" ++
  "        n_batches = 0\n" ++
  "        for i in range(0, len(train_images) - BATCH_SIZE + 1, BATCH_SIZE):\n" ++
  "            x = jax.device_put(shuf_images[i:i+BATCH_SIZE], data_sharding)\n" ++
  "            y = jax.device_put(shuf_labels[i:i+BATCH_SIZE], data_sharding)\n" ++
  (if useAdam then
  "            params, opt_state, loss = train_step(params, opt_state, x, y, lr)\n"
  else if hasMomentum then
  "            params, velocity, loss = train_step(params, velocity, x, y, lr)\n"
  else
  "            params, loss = train_step(params, x, y, lr)\n") ++
  "            epoch_loss += float(loss)\n" ++
  "            n_batches += 1\n\n" ++
  "        correct, total, test_loss = evaluate(params, test_images, test_labels)\n" ++
  "        accuracy = correct / total\n" ++
  "        elapsed = time.time() - t0\n" ++
  "        print(\"[Epoch \" + str(epoch+1) + \"] lr=\" + str(round(float(lr), 6)) +\n" ++
  "              \" Accuracy: \" + str(correct) + \"/\" + str(total) +\n" ++
  "              \" (\" + str(round(accuracy, 4)) + \") Loss: \" + str(round(test_loss, 4)) +\n" ++
  "              \"  [\" + str(round(elapsed, 1)) + \"s]\")\n\n" ++
  "    print(\"\")\n" ++
  "    print(\"Done. Total time: \" + str(round(time.time() - t0, 1)) + \"s\")\n"

private def emitShardingSetup : String :=
  "# ═══════════════════════════════════════════════════════════════════════\n" ++
  "#  Multi-GPU data parallelism\n" ++
  "# ═══════════════════════════════════════════════════════════════════════\n\n" ++
  "devices = jax.devices()\n" ++
  "n_devices = len(devices)\n" ++
  "mesh = Mesh(np.array(devices), axis_names=('batch',))\n" ++
  "data_sharding = NamedSharding(mesh, P('batch'))\n" ++
  "replicated_sharding = NamedSharding(mesh, P())\n\n"

/-- Generate a complete, self-contained JAX training script from a Lean spec. -/
def generate (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind) (dataDir : String) : String :=
  emitHeader spec ++
  emitImports ++
  emitDataLoading ds ++
  emitHelpers spec ++
  emitShardingSetup ++
  emitInitParams spec ++
  emitForward spec ++
  emitLossAndTraining spec cfg ++
  emitMain spec cfg ds dataDir

end JaxCodegen

-- ===========================================================================
-- § 4  Runner
-- ===========================================================================

def findPython : IO String := do
  let r ← IO.Process.output { cmd := "test", args := #["-f", ".venv/bin/python3"] }
  if r.exitCode == 0 then return ".venv/bin/python3"
  return "python3"

def runJax (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind) (dataDir scriptName : String) : IO Unit := do
  IO.println s!"Lean 4 → JAX  {spec.name}"
  IO.println s!"  arch:   {spec.archStr}"
  IO.println s!"  params: {spec.totalParams}"
  IO.println s!"  data:   {dataDir}"
  IO.println ""

  let code := JaxCodegen.generate spec cfg ds dataDir
  let scriptPath := ".lake/build/" ++ scriptName
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile scriptPath code
  IO.println s!"Generated: {scriptPath} ({code.length} chars)"
  IO.println "Running JAX training...\n"

  let python ← findPython
  let child ← IO.Process.spawn {
    cmd := python
    args := #[scriptPath]
    stdout := .piped
    stderr := .piped
    stdin  := .null
  }

  let stdout ← child.stdout.readToEnd
  IO.print stdout

  let stderr ← child.stderr.readToEnd
  let exitCode ← child.wait
  if exitCode != 0 then
    IO.eprintln s!"\nJAX process exited with code {exitCode}"
    IO.eprintln stderr
