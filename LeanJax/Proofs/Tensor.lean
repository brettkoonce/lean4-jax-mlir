/-!
# Tensor Algebra for VJP Proofs

Vectors, matrices, and the operations needed to state and prove
VJP (vector-Jacobian product) correctness for neural network layers.

## Design choices

- **Scalars**: We use `Float` for readability. Algebraic proofs are marked
  `sorry`; correctness holds over ℝ. The ℝ → Float32 gap is bounded
  numerical error (IEEE 754, ~2⁻²⁴ relative per op), orthogonal to the
  question of whether the VJP formulas are mathematically right.

- **Summation**: Axiomatized via `finSum` to keep proofs focused on the
  calculus structure rather than arithmetic induction. Compiles to a loop
  at runtime.

- **Differentiation**: `pdiv` (partial derivative) is axiomatized; the
  multivariable chain rule is stated as `pdiv_comp`. These are theorems of
  real analysis that we take as given.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Types
-- ════════════════════════════════════════════════════════════════

/-- A vector of dimension `n`, indexed by `Fin n`. -/
abbrev Vec (n : Nat) := Fin n → Float

/-- A matrix with `m` rows and `n` columns. -/
abbrev Mat (m n : Nat) := Fin m → Fin n → Float

-- ════════════════════════════════════════════════════════════════
-- § Summation (axiomatized)
-- ════════════════════════════════════════════════════════════════

/-- Σᵢ₌₀ⁿ⁻¹ f(i). A for-loop at runtime; axiomatized for proofs. -/
axiom finSum (n : Nat) (f : Fin n → Float) : Float

/-- Σ(f + g) = Σf + Σg -/
axiom finSum_add (n : Nat) (f g : Fin n → Float) :
    finSum n (fun i => f i + g i) = finSum n f + finSum n g

/-- Σ(c · f) = c · Σf -/
axiom finSum_mul_left (n : Nat) (c : Float) (f : Fin n → Float) :
    finSum n (fun i => c * f i) = c * finSum n f

/-- Finite Fubini: swap order of summation. -/
axiom finSum_swap (m n : Nat) (f : Fin m → Fin n → Float) :
    finSum m (fun i => finSum n (fun j => f i j))
    = finSum n (fun j => finSum m (fun i => f i j))

-- ════════════════════════════════════════════════════════════════
-- § Matrix Operations
-- ════════════════════════════════════════════════════════════════

namespace Mat

/-- Transpose: (Aᵀ)ⱼᵢ = Aᵢⱼ -/
def transpose (A : Mat m n) : Mat n m := fun j i => A i j

/-- Matrix-vector product: (Av)ᵢ = Σⱼ Aᵢⱼ · vⱼ

    For a dense layer's input gradient, this computes Wᵀdy in disguise:
    `mulVec W dy` produces `(grad)ᵢ = Σⱼ Wᵢⱼ · dyⱼ`. -/
noncomputable def mulVec (A : Mat m n) (v : Vec n) : Vec m :=
  fun i => finSum n (fun j => A i j * v j)

/-- Outer product: (u ⊗ v)ᵢⱼ = uᵢ · vⱼ

    The dense layer's weight gradient is exactly the outer product
    of its input with its output cotangent. -/
def outer (u : Vec m) (v : Vec n) : Mat m n :=
  fun i j => u i * v j

/-- Matrix multiplication: (A · B)ᵢₖ = Σⱼ Aᵢⱼ · Bⱼₖ -/
noncomputable def mul (A : Mat m n) (B : Mat n p) : Mat m p :=
  fun i k => finSum n (fun j => A i j * B j k)

/-- Sum rows: (sumRows A)ⱼ = Σᵢ Aᵢⱼ -/
noncomputable def sumRows (A : Mat m n) : Vec n :=
  fun j => finSum m (fun i => A i j)

end Mat

-- ════════════════════════════════════════════════════════════════
-- § Differentiation (axiomatized)
-- ════════════════════════════════════════════════════════════════

/-- ∂fⱼ/∂xᵢ evaluated at x. The partial derivative of the j-th component
    of f with respect to the i-th input. -/
axiom pdiv {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) : Float

/-- **Multivariable chain rule** (real analysis):

    ∂(g ∘ f)ₖ/∂xᵢ = Σⱼ (∂fⱼ/∂xᵢ)(x) · (∂gₖ/∂yⱼ)(f(x))

    This is the foundational fact that makes backpropagation work. -/
axiom pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    finSum n (fun j => pdiv f x i j * pdiv g (f x) j k)

/-- Scalar-valued partial derivative: for f : Vec m → Float, gives ∂f/∂xᵢ.
    Used for loss functions, which output a single scalar. -/
axiom sdiv {m : Nat} (f : Vec m → Float) (x : Vec m) (i : Fin m) : Float

-- ════════════════════════════════════════════════════════════════
-- § VJP Framework
-- ════════════════════════════════════════════════════════════════

/-- A correct VJP (vector-Jacobian product) for `f : Vec m → Vec n`.

    `backward(x, dy)ᵢ = Σⱼ (∂fⱼ/∂xᵢ)(x) · dyⱼ = (Jᵀdy)ᵢ`

    Reverse-mode autodiff in one equation: given an output cotangent dy
    ("how does the loss change per unit change in fⱼ?"), produce the
    input cotangent ("how does the loss change per unit change in xᵢ?").

    The chain rule (`vjp_comp` below) glues these together for a whole
    network: each layer's `backward` is a step of the backprop loop. -/
structure HasVJP {m n : Nat} (f : Vec m → Vec n) where
  backward : Vec m → Vec n → Vec m
  correct : ∀ (x : Vec m) (dy : Vec n) (i : Fin m),
    backward x dy i = finSum n (fun j => pdiv f x i j * dy j)

/-- **Chain rule for VJPs** — the heart of backpropagation.

    If layers `f` and `g` each have correct VJPs, then `g ∘ f` has a
    correct VJP given by composing the backwards in the opposite order:

        back_{g∘f}(x, dy) = back_f(x, back_g(f(x), dy))

    In words: pass the gradient backward through `g`, then through `f`.

    This is the lemma that makes the framework worth having. Once you
    prove VJPs for individual layer types (dense, ReLU, conv, …), you
    get the VJP of any network built from them for free.

    **Proof sketch** (the body is `sorry` for sum-manipulation reasons,
    but the structure is exactly):

      LHS = back_f(x, back_g(f(x), dy))ᵢ                    (definition)
          = Σⱼ ∂fⱼ/∂xᵢ · back_g(f(x), dy)ⱼ                  (by hf.correct)
          = Σⱼ ∂fⱼ/∂xᵢ · (Σₖ ∂gₖ/∂yⱼ · dyₖ)                  (by hg.correct)
          = Σⱼ Σₖ (∂fⱼ/∂xᵢ · ∂gₖ/∂yⱼ) · dyₖ                  (distributivity)
          = Σₖ (Σⱼ ∂fⱼ/∂xᵢ · ∂gₖ/∂yⱼ) · dyₖ                  (Fubini, finSum_swap)
          = Σₖ ∂(g∘f)ₖ/∂xᵢ · dyₖ                            (by pdiv_comp)
          = RHS                                              ✓
-/
noncomputable def vjp_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (g ∘ f) where
  backward := fun x dy => hf.backward x (hg.backward (f x) dy)
  correct := by
    intro x dy i
    sorry

end Proofs
