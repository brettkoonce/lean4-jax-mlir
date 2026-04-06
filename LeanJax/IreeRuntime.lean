/-! Lean FFI bindings for the IREE runtime.

    Links to `libiree_ffi.so` (thin wrapper) + IREE runtime via the Lean shim
    in `ffi/iree_lean_ffi.c`. Exposes:
      - `IreeSession.create` — load a .vmfb, bind to CUDA device
      - `IreeSession.mlpForward` — MLP-specific forward pass (MNIST shape) -/

/-- Opaque handle to an IREE runtime session (module + device). -/
private opaque IreeSessionPointed : NonemptyType
def IreeSession : Type := IreeSessionPointed.type
instance : Nonempty IreeSession := IreeSessionPointed.property

namespace IreeSession

/-- Load a `.vmfb` bytecode module onto the default CUDA device. -/
@[extern "lean_iree_session_create"]
opaque create (path : @& String) : IO IreeSession

/-- Run MNIST-MLP forward pass. Shapes are fixed:
    `x` is `batch×784`, `W0` is `784×512`, `b0` is `512`,
    `W1` is `512×512`, `b1` is `512`, `W2` is `512×10`, `b2` is `10`.
    Returns the logits as a `batch×10` flattened `FloatArray`. -/
@[extern "lean_iree_mlp_forward"]
opaque mlpForward
  (sess : @& IreeSession)
  (x : @& FloatArray)
  (W0 : @& FloatArray) (b0 : @& FloatArray)
  (W1 : @& FloatArray) (b1 : @& FloatArray)
  (W2 : @& FloatArray) (b2 : @& FloatArray)
  (batch : USize) : IO FloatArray

/-- Run one SGD training step. Params packed into a single FloatArray of
    length 669706 in order `W0|b0|W1|b1|W2|b2`. Labels are a ByteArray of
    `4*batch` bytes (int32 LE). Returns new params + loss as a single
    FloatArray of length 669707; `result[669706]` is the loss. -/
@[extern "lean_iree_mlp_train_step"]
opaque mlpTrainStep
  (sess : @& IreeSession)
  (params : @& FloatArray)
  (x : @& FloatArray)
  (y : @& ByteArray)
  (lr : Float)
  (batch : USize) : IO FloatArray

/-- Generic train step. Shapes are packed ByteArrays (see `packShapes`). -/
@[extern "lean_iree_train_step_packed"]
opaque trainStepPacked
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& FloatArray) (shapes : @& ByteArray)
  (x : @& FloatArray) (xShape : @& ByteArray)
  (y : @& ByteArray)
  (lr : Float) (batch : USize) : IO FloatArray

end IreeSession

/- Sizes for the packed-params layout. -/
namespace MlpLayout
def nW0 : Nat := 784 * 512  -- 401408
def nb0 : Nat := 512
def nW1 : Nat := 512 * 512  -- 262144
def nb1 : Nat := 512
def nW2 : Nat := 512 * 10   -- 5120
def nb2 : Nat := 10
def nParams : Nat := nW0 + nb0 + nW1 + nb1 + nW2 + nb2  -- 669706
def lossIdx : Nat := nParams
end MlpLayout

/-- Push a UInt32 as 4 bytes little-endian. -/
private def pushU32 (ba : ByteArray) (v : Nat) : ByteArray := Id.run do
  let mut b := ba
  b := b.push (v % 256).toUInt8
  b := b.push ((v / 256) % 256).toUInt8
  b := b.push ((v / 65536) % 256).toUInt8
  b := b.push ((v / 16777216) % 256).toUInt8
  return b

/-- Pack param shape descriptors: `[nParams, rank0, d0..., rank1, d1..., ...]` as int32 LE. -/
def packShapes (shapes : Array (Array Nat)) : ByteArray := Id.run do
  let mut ba := pushU32 .empty shapes.size
  for shape in shapes do
    ba := pushU32 ba shape.size
    for d in shape do ba := pushU32 ba d
  return ba

/-- Pack a single shape: `[rank, d0, d1, ...]` as int32 LE (for x input). -/
def packXShape (dims : Array Nat) : ByteArray := Id.run do
  let mut ba := pushU32 .empty dims.size
  for d in dims do ba := pushU32 ba d
  return ba

namespace CnnLayout
def paramShapes : Array (Array Nat) := #[
  #[32, 1, 3, 3], #[32],          -- conv0
  #[32, 32, 3, 3], #[32],         -- conv1
  #[6272, 512], #[512],           -- dense0
  #[512, 512], #[512],            -- dense1
  #[512, 10], #[10]               -- dense2
]
def nParams : Nat := 32*1*3*3 + 32 + 32*32*3*3 + 32 + 6272*512 + 512 + 512*512 + 512 + 512*10 + 10
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 784]
end CnnLayout
