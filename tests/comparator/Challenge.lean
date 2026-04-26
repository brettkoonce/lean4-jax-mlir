import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.Attention

open Proofs
open scoped Real

/-! # Challenge file for `leanprover/comparator`

Each `chk_<name>` is a re-statement of a project theorem with `:= by sorry`.
The companion `Solution.lean` discharges each via the corresponding proven
theorem from `LeanMlir.Proofs.*`. comparator confirms (a) the statements
match bit-identically, (b) only `propext, Quot.sound, Classical.choice`
appear in each Solution-side proof's transitive axiom closure, and
(c) the Solution typechecks against Lean's kernel independently of the
elaborator. See `README.md` for the full prereq + run instructions.

The 27 theorems below span the foundation rules + every chapter's headline
Jacobian — enough to verify "zero project axioms" reaches everywhere. The
docstrings give each theorem's mathematical content in regular LaTeX (so
non-Lean readers can follow the math) and a one-line note on its role in
the proof tree.
-/

-- Foundation: structural calculus rules ────────────────────────────

/-- **Chain rule** (foundation):
$\frac{\partial (g \circ f)_k}{\partial x_i} = \sum_j \frac{\partial f_j}{\partial x_i} \cdot \frac{\partial g_k}{\partial f_j}$

Used by every backward pass that composes two functions. The structural
glue that makes the whole proof tree compositional. -/
theorem chk_pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k := by sorry

/-- **Sum rule** / linearity (foundation):
$\frac{\partial (f+g)_j}{\partial x_i} = \frac{\partial f_j}{\partial x_i} + \frac{\partial g_j}{\partial x_i}$

Underpins additive fan-in (residual connections, multi-path attention). -/
theorem chk_pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j := by sorry

/-- **Product rule** / Leibniz (foundation):
$\frac{\partial (f \cdot g)_j}{\partial x_i} = \frac{\partial f_j}{\partial x_i} \cdot g_j + f_j \cdot \frac{\partial g_j}{\partial x_i}$

Underpins multiplicative fan-in (squeeze-and-excitation gating, attention
weights × values). -/
theorem chk_pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j := by sorry

/-- **Identity Jacobian** (foundation):
$\frac{\partial \text{id}(x)_j}{\partial x_i} = \delta_{ij}$

Building block for skip connections and identity bridges in residuals. -/
theorem chk_pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 := by sorry

/-- **Constant has zero Jacobian** (foundation):
$\frac{\partial c}{\partial x_i} = 0$

Discharges bias terms when isolating weight gradients. -/
theorem chk_pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 := by sorry

/-- **Reindex / gather Jacobian** (foundation):
$\frac{\partial y_{\sigma(k)}}{\partial x_i} = \delta_{i, \sigma(k)}$

Backbone of every reshape/transpose/slice operation
(`Mat.flatten`, `Mat.unflatten`, attention head extraction). -/
theorem chk_pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 := by sorry

/-- **Linearity over finite sums** (foundation):
$\frac{\partial}{\partial x_i} \sum_{s \in S} f_s(y)_j = \sum_{s \in S} \frac{\partial f_s(y)_j}{\partial x_i}$

Extension of `pdiv_add` to arbitrary index sets. Load-bearing for
conv2d / depthwise weight gradients (their inner sums over kernel windows). -/
theorem chk_pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j := by sorry

/-- **Row-independence for matrices** (foundation, was the last surviving
Mat-level axiom in earlier drafts; now proved):
A function that applies a vector-to-vector $g$ row-wise to a matrix has
a block-diagonal Jacobian: only entries within the same row see each other.

Lifts any proved $\text{Vec} \to \text{Vec}$ result to a row-wise matrix
operation (per-token `softmax`, per-token `layerNorm`, per-token `gelu`,
per-token dense). -/
theorem chk_pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 := by sorry

-- Mat-level structural rules ───────────────────────────────────────

/-- **Matrix-level chain rule**: same form as the scalar `pdiv_comp`,
applied to functions $\text{Mat} \to \text{Mat}$ via the
`Mat.flatten`/`Mat.unflatten` bijection.

Glues the four pieces of attention's SDPA backward (matmul → scale →
rowSoftmax → matmul). -/
theorem chk_pdivMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)))
    (i : Fin a) (j : Fin b) (k : Fin e) (l : Fin f) :
    pdivMat (G ∘ F) A i j k l =
    ∑ p : Fin c, ∑ q : Fin d,
      pdivMat F A i j p q * pdivMat G (F A) p q k l := by sorry

/-- **Matmul Jacobian, left factor fixed**:
$\frac{\partial (C \cdot B)_{k,l}}{\partial B_{i,j}} = \delta_{l,j} \cdot C_{k,i}$

The "input gradient through a matmul" — building block for SDPA's
score computation $Q \cdot K^T$ and attention output $\text{weights} \cdot V$. -/
theorem chk_pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 := by sorry

/-- **Scalar-scale Jacobian**:
$\frac{\partial (s \cdot M)_{k,l}}{\partial M_{i,j}} = s \cdot \delta_{i,k} \delta_{j,l}$

Diagonal Jacobian for elementwise scaling. Used by attention's
$1/\sqrt{d_k}$ scale factor. -/
theorem chk_pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 := by sorry

/-- **Transpose Jacobian** (a permutation):
$\frac{\partial M^T_{k,l}}{\partial M_{i,j}} = \delta_{j,k} \delta_{i,l}$

Used by attention's $K^T$ in the score computation
$\text{scores} = Q \cdot K^T$. -/
theorem chk_pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 := by sorry

-- Ch 3 MLP ──────────────────────────────────────────────────────────

/-- **Dense layer Jacobian wrt input**:
$\frac{\partial (Wx + b)_j}{\partial x_i} = W_{i,j}$

The simplest non-trivial Jacobian; the entry point for ML derivatives.
Every dense layer's input gradient is "transpose the weights and matmul." -/
theorem chk_pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j := by sorry

/-- **Dense layer Jacobian wrt weight** (with the weight matrix flattened
to a vector so `pdiv` applies):
$\frac{\partial (W \cdot x + b)_j}{\partial W_{i, j'}} = \delta_{j, j'} \cdot x_i$

Each weight $W_{i,j'}$ contributes to exactly one output coordinate ($j'$)
with magnitude equal to the corresponding input ($x_i$). -/
theorem chk_pdiv_dense_W {m n : Nat} (b : Vec n) (x : Vec m) (W : Mat m n)
    (i : Fin m) (j' : Fin n) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
         (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if j = j' then x i else 0 := by sorry

/-- **Dense layer Jacobian wrt bias**:
$\frac{\partial (W \cdot x + b)_j}{\partial b_i} = \delta_{i, j}$

Bias enters the output diagonally; this is why bias gradients are
"just sum the cotangent." -/
theorem chk_pdiv_dense_b {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (i j : Fin n) :
    pdiv (fun b' : Vec n => dense W b' x) b i j = if i = j then 1 else 0 := by sorry

/-- **The outer product is the dense weight gradient**:
$dW_{i,j} = x_i \cdot dy_j$

The famous "outer product of input and gradient" identity. Combines
`pdiv_dense_W` with the cotangent contraction; the resulting formula
is what every ML framework emits for dense backward. -/
theorem chk_dense_weight_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) (j : Fin n) :
    Mat.outer x dy i j =
      ∑ k : Fin n,
        pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
             (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k := by sorry

/-- **The cotangent itself is the dense bias gradient**:
$db_i = dy_i$

The simplest gradient identity in deep learning. Combines `pdiv_dense_b`
(diagonal Jacobian) with the cotangent contraction. -/
theorem chk_dense_bias_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin n) :
    dy i =
      ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j := by sorry

-- Ch 5 BN ───────────────────────────────────────────────────────────

/-- **BN affine step (scale + shift) Jacobian**:
$\frac{\partial (\gamma v + \beta)_j}{\partial v_i} = \gamma \cdot \delta_{i,j}$

The "easy half" of BN's backward — diagonal scaling by the learnable
$\gamma$ parameter. -/
theorem chk_pdiv_bnAffine (n : Nat) (γ β : ℝ) (v : Vec n) (i j : Fin n) :
    pdiv (bnAffine n γ β) v i j =
      if i = j then γ else 0 := by sorry

/-- **BN centering step Jacobian**:
$\frac{\partial (x_j - \mu(x))}{\partial x_i} = \delta_{i,j} - \frac{1}{n}$

Subtracting the mean introduces the universal $-1/n$ term — every
output entry gets a tiny negative contribution from every input entry. -/
theorem chk_pdiv_bnCentered (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (bnCentered n) x i j =
      (if i = j then (1 : ℝ) else 0) - 1 / (n : ℝ) := by sorry

/-- **BN inverse-stddev broadcast Jacobian** (the hard one):
$\frac{\partial \text{istd}(x)}{\partial x_i} = -\text{istd}(x)^3 \cdot \frac{x_i - \mu}{n}$

Chain rule through $\sqrt{\sigma^2 + \varepsilon}$ + reciprocal +
the centered-variance expression. The smoothness condition $\varepsilon > 0$
keeps the derivative defined everywhere. -/
theorem chk_pdiv_bnIstdBroadcast (n : Nat) (ε : ℝ) (hε : 0 < ε) (x : Vec n)
    (i j : Fin n) :
    pdiv (bnIstdBroadcast n ε) x i j =
      -(bnIstd n x ε)^3 * (x i - bnMean n x) / (n : ℝ) := by sorry

/-- **The famous BN three-term Jacobian** (the consolidated formula
every ML framework hard-codes):
$\frac{\partial \hat{x}_j}{\partial x_i} = \frac{\text{istd}}{n} \cdot \left( n \cdot \delta_{i,j} - 1 - \hat{x}_i \cdot \hat{x}_j \right)$

Derives from `pdiv_bnCentered` + `pdiv_bnIstdBroadcast` via the product
rule. The "consolidated" formula collapses what would be three reductions
(over $\mu$, $\sigma^2$, and $\hat{x}$) into one closed-form expression
that an ML framework can emit cheaply. -/
theorem chk_pdiv_bnNormalize (n : Nat) (ε : ℝ) (hε : 0 < ε)
    (x : Vec n) (i j : Fin n) :
    pdiv (bnNormalize n ε) x i j =
      bnIstd n x ε / (n : ℝ) *
        ((n : ℝ) * (if i = j then 1 else 0) - 1 - bnXhat n ε x i * bnXhat n ε x j) := by sorry

-- Ch 9 LayerNorm + GELU ─────────────────────────────────────────────

/-- **GELU activation Jacobian** (diagonal):
$\frac{\partial \text{gelu}(x)_j}{\partial x_i} = \delta_{i,j} \cdot \text{gelu}'(x_i)$

GELU is a smooth approximation of ReLU (uses $\tanh$ internally), so its
Jacobian is genuinely diagonal — no kink, no convention pick. -/
theorem chk_pdiv_gelu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gelu n) x i j =
    if i = j then geluScalarDeriv (x i) else 0 := by sorry

-- Ch 10 Attention ───────────────────────────────────────────────────

/-- **Softmax Jacobian** (rank-1 correction to a diagonal):
$\frac{\partial \text{softmax}(z)_j}{\partial z_i} = p_j \cdot (\delta_{i,j} - p_i)$

where $p = \text{softmax}(z)$. The "rank-1 correction" structure is
universal in attention and classifier backwards. -/
theorem chk_pdiv_softmax (c : Nat) (z : Vec c) (i j : Fin c) :
    pdiv (softmax c) z i j =
    softmax c z j * ((if i = j then 1 else 0) - softmax c z i) := by sorry

/-- **Softmax + cross-entropy gradient** (the famous ML identity):
$\frac{\partial L}{\partial z_j} = \text{softmax}(z)_j - \text{onehot}(\text{label})_j$

The "predictions minus labels" formula — the workhorse of every
classification training step. The derivation chains the softmax
Jacobian with $-\log$. -/
theorem chk_softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0
    = softmax c logits j - oneHot c label j := by sorry

/-- **SDPA backward wrt Q**: the Q-input gradient of scaled dot-product
attention $\text{softmax}(QK^T / \sqrt{d}) V$.

Composes four already-proved building blocks (matmul-right-const,
scalarScale, rowSoftmax, matmul-right-const) via `pdivMat_comp`. -/
theorem chk_sdpa_back_Q_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_Q n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun Q' => sdpa n d Q' K V) Q i j k l * dOut k l := by sorry

/-- **SDPA backward wrt K**: same structural composition as Q but routes
through the transpose theorem ($K$ enters the score computation as $K^T$). -/
theorem chk_sdpa_back_K_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_K n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun K' => sdpa n d Q K' V) K i j k l * dOut k l := by sorry

/-- **SDPA backward wrt V**: simpler than Q/K because $V$ only appears
in the final $\text{weights} \cdot V$ matmul (not inside the softmax). -/
theorem chk_sdpa_back_V_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_V n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun V' => sdpa n d Q K V') V i j k l * dOut k l := by sorry
