import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.Attention

open Proofs
open scoped Real

-- Foundation ────────────────────────────────────────────────────────

theorem chk_pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k :=
  pdiv_comp f g x hf hg i k

theorem chk_pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j :=
  pdiv_add f g x hf hg i j

theorem chk_pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j :=
  pdiv_mul f g x hf hg i j

theorem chk_pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 :=
  pdiv_id x i j

theorem chk_pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 :=
  pdiv_const c x i j

theorem chk_pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 :=
  pdiv_reindex σ x i j

theorem chk_pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j :=
  pdiv_finset_sum S f x hdiff i j

theorem chk_pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 :=
  pdivMat_rowIndep g h_g_diff A i j k l

-- Mat-level ─────────────────────────────────────────────────────────

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
      pdivMat F A i j p q * pdivMat G (F A) p q k l :=
  pdivMat_comp F G A hF_diff hG_diff i j k l

theorem chk_pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 :=
  pdivMat_matmul_left_const C B i j k l

theorem chk_pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 :=
  pdivMat_scalarScale s A i j k l

theorem chk_pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 :=
  pdivMat_transpose A i j k l

-- Ch 3 MLP ──────────────────────────────────────────────────────────

theorem chk_pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j :=
  pdiv_dense W b x i j

theorem chk_pdiv_dense_W {m n : Nat} (b : Vec n) (x : Vec m) (W : Mat m n)
    (i : Fin m) (j' : Fin n) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
         (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if j = j' then x i else 0 :=
  pdiv_dense_W b x W i j' j

theorem chk_pdiv_dense_b {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (i j : Fin n) :
    pdiv (fun b' : Vec n => dense W b' x) b i j = if i = j then 1 else 0 :=
  pdiv_dense_b W b x i j

theorem chk_dense_weight_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) (j : Fin n) :
    Mat.outer x dy i j =
      ∑ k : Fin n,
        pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
             (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k :=
  dense_weight_grad_correct W b x dy i j

theorem chk_dense_bias_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin n) :
    dy i =
      ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j :=
  dense_bias_grad_correct W b x dy i

-- Ch 5 BN ───────────────────────────────────────────────────────────

theorem chk_pdiv_bnAffine (n : Nat) (γ β : ℝ) (v : Vec n) (i j : Fin n) :
    pdiv (bnAffine n γ β) v i j =
      if i = j then γ else 0 :=
  pdiv_bnAffine n γ β v i j

theorem chk_pdiv_bnCentered (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (bnCentered n) x i j =
      (if i = j then (1 : ℝ) else 0) - 1 / (n : ℝ) :=
  pdiv_bnCentered n x i j

theorem chk_pdiv_bnIstdBroadcast (n : Nat) (ε : ℝ) (hε : 0 < ε) (x : Vec n)
    (i j : Fin n) :
    pdiv (bnIstdBroadcast n ε) x i j =
      -(bnIstd n x ε)^3 * (x i - bnMean n x) / (n : ℝ) :=
  pdiv_bnIstdBroadcast n ε hε x i j

theorem chk_pdiv_bnNormalize (n : Nat) (ε : ℝ) (hε : 0 < ε)
    (x : Vec n) (i j : Fin n) :
    pdiv (bnNormalize n ε) x i j =
      bnIstd n x ε / (n : ℝ) *
        ((n : ℝ) * (if i = j then 1 else 0) - 1 - bnXhat n ε x i * bnXhat n ε x j) :=
  pdiv_bnNormalize n ε hε x i j

-- Ch 9 LayerNorm + GELU ─────────────────────────────────────────────

theorem chk_pdiv_gelu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gelu n) x i j =
    if i = j then geluScalarDeriv (x i) else 0 :=
  pdiv_gelu n x i j

-- Ch 10 Attention ───────────────────────────────────────────────────

theorem chk_pdiv_softmax (c : Nat) (z : Vec c) (i j : Fin c) :
    pdiv (softmax c) z i j =
    softmax c z j * ((if i = j then 1 else 0) - softmax c z i) :=
  pdiv_softmax c z i j

theorem chk_softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0
    = softmax c logits j - oneHot c label j :=
  softmaxCE_grad c logits label j

theorem chk_sdpa_back_Q_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_Q n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun Q' => sdpa n d Q' K V) Q i j k l * dOut k l :=
  sdpa_back_Q_correct n d Q K V dOut i j

theorem chk_sdpa_back_K_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_K n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun K' => sdpa n d Q K' V) K i j k l * dOut k l :=
  sdpa_back_K_correct n d Q K V dOut i j

theorem chk_sdpa_back_V_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_V n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun V' => sdpa n d Q K V') V i j k l * dOut k l :=
  sdpa_back_V_correct n d Q K V dOut i j

-- Public correctness theorems for canonical-witness defs ────────────

theorem chk_relu_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  relu_has_vjp_correct n x dy i

theorem chk_mlp_has_vjp_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (dy : Vec d₃) (i : Fin d₀) :
    (mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).backward x dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  mlp_has_vjp_correct W₀ b₀ W₁ b₁ W₂ b₂ x dy i

theorem chk_maxPool2_has_vjp3_correct {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    (maxPool2_has_vjp3 (c := c) (h := h) (w := w)).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  maxPool2_has_vjp3_correct x dy ci hi wi

theorem chk_depthwise_has_vjp3_correct {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    (depthwise_has_vjp3 (h := h) (w := w) W b).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  depthwise_has_vjp3_correct W b x dy ci hi wi

theorem chk_residual_has_vjp_correct {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (residual_has_vjp f hf_diff hf).backward x dy i =
    ∑ j : Fin n, pdiv (residual f) x i j * dy j :=
  residual_has_vjp_correct f hf_diff hf x dy i

theorem chk_residualProj_has_vjp_correct {m n : Nat}
    (proj f : Vec m → Vec n)
    (hproj_diff : Differentiable ℝ proj) (hf_diff : Differentiable ℝ f)
    (hproj : HasVJP proj) (hf : HasVJP f)
    (x : Vec m) (dy : Vec n) (i : Fin m) :
    (residualProj_has_vjp proj f hproj_diff hf_diff hproj hf).backward x dy i =
    ∑ j : Fin n, pdiv (residualProj proj f) x i j * dy j :=
  residualProj_has_vjp_correct proj f hproj_diff hf_diff hproj hf x dy i

theorem chk_seBlock_has_vjp_correct {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (seBlock_has_vjp gate hg_diff hg).backward x dy i =
    ∑ j : Fin n, pdiv (seBlock gate) x i j * dy j :=
  seBlock_has_vjp_correct gate hg_diff hg x dy i

theorem chk_gelu_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (gelu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (gelu n) x i j * dy j :=
  gelu_has_vjp_correct n x dy i

theorem chk_layerNorm_has_vjp_correct (n : Nat) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (layerNorm_has_vjp n ε γ β hε).backward x dy i =
    ∑ j : Fin n, pdiv (layerNormForward n ε γ β) x i j * dy j :=
  layerNorm_has_vjp_correct n ε γ β hε x dy i

theorem chk_mhsa_has_vjp_mat_correct (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) (dY : Mat N (heads * d_head))
    (i : Fin N) (j : Fin (heads * d_head)) :
    (mhsa_has_vjp_mat N heads d_head Wq Wk Wv Wo bq bk bv bo).backward X dY i j =
    ∑ k : Fin N, ∑ l : Fin (heads * d_head),
      pdivMat (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo)
              X i j k l * dY k l :=
  mhsa_has_vjp_mat_correct N heads d_head Wq Wk Wv Wo bq bk bv bo X dY i j

theorem chk_transformerBlock_has_vjp_mat_correct
    (N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) (dY : Mat N (heads * d_head))
    (i : Fin N) (j : Fin (heads * d_head)) :
    (transformerBlock_has_vjp_mat N heads d_head mlpDim ε γ1 β1 hε
        Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward X dY i j =
    ∑ k : Fin N, ∑ l : Fin (heads * d_head),
      pdivMat (transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
              X i j k l * dY k l :=
  transformerBlock_has_vjp_mat_correct N heads d_head mlpDim ε γ1 β1 hε
    Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X dY i j
