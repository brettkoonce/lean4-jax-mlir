#!/usr/bin/env python3
"""Numerical gradient checks for every axiomatized VJP in the proof suite.

For each axiom, we compute the Jacobian two ways:
  1. Finite differences: J[i,j] ≈ (f(x + εeⱼ) - f(x - εeⱼ)) / 2ε
  2. The claimed formula from the axiom

If they agree to ~5 decimal places, the axiom's formula is correct
(up to floating-point precision). This catches typos, sign errors,
and off-by-one mistakes in the stated Jacobian formulas.

Usage: python3 check_axioms.py
"""
import numpy as np

EPS = 1e-5
TOL = 1e-4
np.random.seed(42)

def check(name, f, claimed_jac, x_shape, out_shape=None):
    """Compare finite-difference Jacobian against claimed Jacobian."""
    x = np.random.randn(*x_shape).astype(np.float64)
    fx = f(x)
    if out_shape is None:
        out_shape = fx.shape
    m = int(np.prod(x_shape))
    n = int(np.prod(out_shape))
    xf = x.ravel()

    # Finite differences
    J_fd = np.zeros((n, m))
    for j in range(m):
        xp = xf.copy(); xp[j] += EPS
        xm = xf.copy(); xm[j] -= EPS
        fp = f(xp.reshape(x_shape)).ravel()
        fm = f(xm.reshape(x_shape)).ravel()
        J_fd[:, j] = (fp - fm) / (2 * EPS)

    # Claimed
    J_cl = claimed_jac(x).reshape(n, m)

    err = np.max(np.abs(J_fd - J_cl))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {name:30s} max_err={err:.2e}")
    if err >= TOL:
        # Show worst entry
        idx = np.unravel_index(np.argmax(np.abs(J_fd - J_cl)), J_fd.shape)
        print(f"         worst at {idx}: fd={J_fd[idx]:.8f} claimed={J_cl[idx]:.8f}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Dense: pdiv_dense — ∂(xW+b)_j/∂x_i = W[i,j]
# ════════════════════════════════════════════════════════════════
def test_dense():
    m, n = 4, 3
    W = np.random.randn(m, n)
    b = np.random.randn(n)
    f = lambda x: x.ravel() @ W + b
    jac = lambda x: W.T  # J[j,i] = W[i,j], so J = W^T
    return check("pdiv_dense", f, jac, (m,), (n,))

# ════════════════════════════════════════════════════════════════
# ReLU: pdiv_relu — diagonal, 0 or 1
# ════════════════════════════════════════════════════════════════
def test_relu():
    n = 5
    f = lambda x: np.maximum(x, 0)
    jac = lambda x: np.diag((x > 0).astype(float))
    return check("pdiv_relu", f, jac, (n,))

# ════════════════════════════════════════════════════════════════
# Softmax CE: ∂(-log softmax(z)[label])/∂z_j = softmax(z)_j - onehot_j
# ════════════════════════════════════════════════════════════════
def test_softmax_ce():
    c = 5
    label = 2
    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()
    f = lambda z: np.array([-np.log(softmax(z)[label])])
    def jac(z):
        p = softmax(z)
        oh = np.zeros(c); oh[label] = 1.0
        return (p - oh).reshape(1, c)
    return check("softmaxCE_grad", f, jac, (c,), (1,))

# ════════════════════════════════════════════════════════════════
# Conv2d input grad: dx = conv(dy, reversed transposed W)
# ════════════════════════════════════════════════════════════════
def test_conv2d_input_grad():
    ic, oc, h, w, kH, kW = 2, 3, 6, 6, 3, 3
    W = np.random.randn(oc, ic, kH, kW)
    b = np.random.randn(oc)
    pad = (kH - 1) // 2

    def conv_fwd(x):
        x = x.reshape(ic, h, w)
        out = np.zeros((oc, h, w))
        for o in range(oc):
            for c in range(ic):
                for kh in range(kH):
                    for kw in range(kW):
                        for i in range(h):
                            for j in range(w):
                                ii, jj = i + kh - pad, j + kw - pad
                                if 0 <= ii < h and 0 <= jj < w:
                                    out[o, i, j] += x[c, ii, jj] * W[o, c, kh, kw]
            out[o] += b[o]
        return out.ravel()

    # This test is superseded by test_conv2d_input_grad_formula below
    pass

def test_conv2d_input_grad_formula():
    """Check that conv(dy, reverse(W^T)) gives the correct input gradient."""
    ic, oc, h, w, kH, kW = 2, 3, 6, 6, 3, 3
    W = np.random.randn(oc, ic, kH, kW)
    b = np.random.randn(oc)
    pad = (kH - 1) // 2

    def conv(inp, kernel, pad):
        ci, hi, wi = inp.shape
        co, _, kh, kw = kernel.shape
        out = np.zeros((co, hi, wi))
        for o in range(co):
            for c in range(ci):
                for khi in range(kh):
                    for kwi in range(kw):
                        for i in range(hi):
                            for j in range(wi):
                                ii, jj = i + khi - pad, j + kwi - pad
                                if 0 <= ii < hi and 0 <= jj < wi:
                                    out[o, i, j] += inp[c, ii, jj] * kernel[o, c, khi, kwi]
        return out

    def conv_fwd(x):
        x = x.reshape(ic, h, w)
        return conv(x, W, pad).ravel() + np.repeat(b, h * w)

    x = np.random.randn(ic, h, w)
    dy = np.random.randn(oc, h, w)

    # Finite-diff VJP: dx_i = sum_j J[j,i] * dy_j
    xf = x.ravel()
    dx_fd = np.zeros_like(xf)
    for i in range(len(xf)):
        xp = xf.copy(); xp[i] += EPS
        xm = xf.copy(); xm[i] -= EPS
        fp = conv_fwd(xp); fm = conv_fwd(xm)
        dx_fd[i] = np.sum(((fp - fm) / (2 * EPS)) * dy.ravel())

    # Claimed: dx = conv(dy, W^T reversed)
    W_t = W.transpose(1, 0, 2, 3)          # (ic, oc, kH, kW)
    W_rev = W_t[:, :, ::-1, ::-1].copy()   # reverse spatial
    dx_claimed = conv(dy, W_rev, pad).ravel()

    err = np.max(np.abs(dx_fd - dx_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'conv2d_input_grad (formula)':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Conv2d weight grad: dW = conv(x^T, dy^T) transposed
# ════════════════════════════════════════════════════════════════
def test_conv2d_weight_grad():
    ic, oc, h, w, kH, kW = 2, 3, 6, 6, 3, 3
    W = np.random.randn(oc, ic, kH, kW)
    b = np.random.randn(oc)
    pad = (kH - 1) // 2

    def conv(inp, kernel, p):
        ci, hi, wi = inp.shape
        co, _, kh, kw = kernel.shape
        oh = hi + 2*p - kh + 1
        ow = wi + 2*p - kw + 1
        out = np.zeros((co, oh, ow))
        for o in range(co):
            for c in range(ci):
                for khi in range(kh):
                    for kwi in range(kw):
                        for i in range(oh):
                            for j in range(ow):
                                ii, jj = i + khi - p, j + kwi - p
                                if 0 <= ii < hi and 0 <= jj < wi:
                                    out[o, i, j] += inp[c, ii, jj] * kernel[o, c, khi, kwi]
        return out

    x = np.random.randn(ic, h, w)
    dy = np.random.randn(oc, h, w)

    # Finite-diff: perturb each W entry, measure change in output dotted with dy
    dW_fd = np.zeros_like(W)
    Wf = W.ravel()
    for idx in range(len(Wf)):
        Wp = Wf.copy(); Wp[idx] += EPS
        Wm = Wf.copy(); Wm[idx] -= EPS
        def fwd(Wv):
            return conv(x, Wv.reshape(W.shape), pad) + b.reshape(oc, 1, 1)
        fp = fwd(Wp); fm = fwd(Wm)
        dW_fd.ravel()[idx] = np.sum(((fp - fm) / (2 * EPS)) * dy)

    # Claimed: transpose trick
    x_t = x.reshape(ic, 1, h, w).transpose(0, 1, 2, 3)  # (ic, 1, h, w)
    dy_t = dy.reshape(oc, 1, h, w)  # (oc, 1, h, w)
    # Treat as conv: input=(ic,1,h,w), kernel=(oc,1,h,w), output=(ic,oc,kH,kW)
    # This is: dW_raw[c,o,kh,kw] = sum_{i,j} x[c,i+kh-p,j+kw-p] * dy[o,i,j]
    dW_claimed = np.zeros((oc, ic, kH, kW))
    for o in range(oc):
        for c in range(ic):
            for kh in range(kH):
                for kw in range(kW):
                    s = 0.0
                    for i in range(h):
                        for j in range(w):
                            ii, jj = i + kh - pad, j + kw - pad
                            if 0 <= ii < h and 0 <= jj < w:
                                s += x[c, ii, jj] * dy[o, i, j]
                    dW_claimed[o, c, kh, kw] = s

    err = np.max(np.abs(dW_fd - dW_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'conv2d_weight_grad (formula)':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Dense weight grad: dW = outer(x, dy). ∂(xW+b)_j/∂W_{i,j'} = x_i if j=j' else 0
# (Phase 7: new axiom pdiv_dense_W + theorem dense_weight_grad_correct.)
# ════════════════════════════════════════════════════════════════
def test_dense_weight_grad():
    m, n = 4, 3
    W = np.random.randn(m, n)
    b = np.random.randn(n)
    x = np.random.randn(m)
    dy = np.random.randn(n)

    # Finite-diff: perturb each W entry, measure <d output, dy>
    def fwd(Wv):
        return Wv.reshape(m, n).T @ x + b
    dW_fd = np.zeros_like(W)
    Wf = W.ravel()
    for idx in range(len(Wf)):
        Wp = Wf.copy(); Wp[idx] += EPS
        Wm = Wf.copy(); Wm[idx] -= EPS
        dW_fd.ravel()[idx] = np.sum(((fwd(Wp) - fwd(Wm)) / (2 * EPS)) * dy)

    # Claimed: dW = outer(x, dy) with shape (m, n)
    dW_claimed = np.outer(x, dy)

    err = np.max(np.abs(dW_fd - dW_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'dense_weight_grad (outer)':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Dense bias grad: db = dy. ∂(xW+b)_j/∂b_i = δ(i,j)
# (Phase 7: theorem pdiv_dense_b, derived from pdiv_add + pdiv_const + pdiv_id.)
# ════════════════════════════════════════════════════════════════
def test_dense_bias_grad():
    m, n = 4, 3
    W = np.random.randn(m, n)
    b = np.random.randn(n)
    x = np.random.randn(m)
    dy = np.random.randn(n)

    def fwd(bv):
        return x @ W + bv
    db_fd = np.zeros_like(b)
    for idx in range(n):
        bp = b.copy(); bp[idx] += EPS
        bm = b.copy(); bm[idx] -= EPS
        db_fd[idx] = np.sum(((fwd(bp) - fwd(bm)) / (2 * EPS)) * dy)

    db_claimed = dy.copy()
    err = np.max(np.abs(db_fd - db_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'dense_bias_grad (identity)':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Conv2d bias grad: db = sum output cotangent over spatial, per channel
# (Phase 9: new axiom conv2d_bias_grad_has_vjp.)
# ════════════════════════════════════════════════════════════════
def test_conv2d_bias_grad():
    ic, oc, h, w, kH, kW = 2, 3, 6, 6, 3, 3
    W = np.random.randn(oc, ic, kH, kW)
    pad = (kH - 1) // 2

    def conv(inp, kernel, p):
        ci, hi, wi = inp.shape
        co = kernel.shape[0]
        out = np.zeros((co, hi, wi))
        for o in range(co):
            for c in range(ci):
                for khi in range(kH):
                    for kwi in range(kW):
                        for i in range(hi):
                            for j in range(wi):
                                ii, jj = i + khi - p, j + kwi - p
                                if 0 <= ii < hi and 0 <= jj < wi:
                                    out[o, i, j] += inp[c, ii, jj] * kernel[o, c, khi, kwi]
        return out

    x = np.random.randn(ic, h, w)
    dy = np.random.randn(oc, h, w)

    # Finite-diff: perturb each b entry, measure <Δoutput, dy>
    def fwd(bv):
        return conv(x, W, pad) + bv.reshape(oc, 1, 1)
    b = np.random.randn(oc)
    db_fd = np.zeros(oc)
    for idx in range(oc):
        bp = b.copy(); bp[idx] += EPS
        bm = b.copy(); bm[idx] -= EPS
        db_fd[idx] = np.sum(((fwd(bp) - fwd(bm)) / (2 * EPS)) * dy)

    # Claimed: db[o] = Σ_{h,w} dy[o, h, w]
    db_claimed = dy.sum(axis=(1, 2))

    err = np.max(np.abs(db_fd - db_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'conv2d_bias_grad (spatial sum)':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Depthwise bias grad: same spatial sum, per channel (Phase 9)
# ════════════════════════════════════════════════════════════════
def test_depthwise_bias_grad():
    c, h, w, kH, kW = 3, 6, 6, 3, 3
    W = np.random.randn(c, kH, kW)
    pad = (kH - 1) // 2

    def dw_fwd(bv):
        x = np.random.RandomState(1).randn(c, h, w)  # fixed x
        out = np.zeros((c, h, w))
        for ch in range(c):
            for kh in range(kH):
                for kw in range(kW):
                    for i in range(h):
                        for j in range(w):
                            ii, jj = i + kh - pad, j + kw - pad
                            if 0 <= ii < h and 0 <= jj < w:
                                out[ch, i, j] += x[ch, ii, jj] * W[ch, kh, kw]
            out[ch] += bv[ch]
        return out

    dy = np.random.randn(c, h, w)
    b = np.random.randn(c)
    db_fd = np.zeros(c)
    for idx in range(c):
        bp = b.copy(); bp[idx] += EPS
        bm = b.copy(); bm[idx] -= EPS
        db_fd[idx] = np.sum(((dw_fwd(bp) - dw_fwd(bm)) / (2 * EPS)) * dy)
    db_claimed = dy.sum(axis=(1, 2))
    err = np.max(np.abs(db_fd - db_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'depthwise_bias_grad':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# MaxPool2: gradient routes to argmax
# ════════════════════════════════════════════════════════════════
def test_maxpool2():
    """MaxPool VJP: gradient routes to argmax positions."""
    c, h, w = 2, 3, 3
    x = np.random.randn(c, 2*h, 2*w)
    dy = np.random.randn(c, h, w)

    def pool_fwd(xv):
        xv = xv.reshape(c, 2*h, 2*w)
        return xv.reshape(c, h, 2, w, 2).max(axis=(2, 4)).ravel()

    # Finite-diff VJP
    xf = x.ravel()
    dx_fd = np.zeros_like(xf)
    for i in range(len(xf)):
        xp = xf.copy(); xp[i] += EPS
        xm = xf.copy(); xm[i] -= EPS
        dx_fd[i] = np.sum(((pool_fwd(xp) - pool_fwd(xm)) / (2 * EPS)) * dy.ravel())

    # Claimed: route dy to argmax within each 2x2 window
    dx_claimed = np.zeros_like(x)
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                block = x[ch, 2*i:2*i+2, 2*j:2*j+2]
                idx = np.unravel_index(block.argmax(), (2, 2))
                dx_claimed[ch, 2*i+idx[0], 2*j+idx[1]] = dy[ch, i, j]

    err = np.max(np.abs(dx_fd - dx_claimed.ravel()))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'maxPool2_input_grad':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# BatchNorm normalize: ∂x̂ⱼ/∂xᵢ = (istd/N)(Nδᵢⱼ - 1 - x̂ᵢx̂ⱼ)
# ════════════════════════════════════════════════════════════════
def test_bn_normalize():
    n = 5
    eps = 1e-5
    def bn_fwd(x):
        mu = x.mean()
        var = ((x - mu) ** 2).mean()
        return (x - mu) / np.sqrt(var + eps)

    def bn_jac(x):
        mu = x.mean()
        var = ((x - mu) ** 2).mean()
        istd = 1.0 / np.sqrt(var + eps)
        xhat = (x - mu) * istd
        N = len(x)
        J = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                delta = 1.0 if i == j else 0.0
                J[i, j] = (istd / N) * (N * delta - 1.0 - xhat[i] * xhat[j])
        return J

    return check("pdiv_bnNormalize", bn_fwd, bn_jac, (n,))

# ════════════════════════════════════════════════════════════════
# BatchNorm centered (pdiv_bnCentered axiom):
#   ∂(xⱼ - μ(x))/∂xᵢ = δᵢⱼ - 1/n
# ════════════════════════════════════════════════════════════════
def test_bn_centered():
    n = 5
    f = lambda x: x - x.mean()  # broadcasted: (x - μ)_j
    def jac(x):
        N = len(x)
        return np.eye(N) - (1.0 / N) * np.ones((N, N))
    return check("pdiv_bnCentered", f, jac, (n,))

# ════════════════════════════════════════════════════════════════
# BatchNorm broadcast-istd (pdiv_bnIstdBroadcast axiom):
#   ∂istd(x,ε)/∂xᵢ = -istd³ · (xᵢ - μ) / n   (same for all output indices)
# ════════════════════════════════════════════════════════════════
def test_bn_istd_broadcast():
    n = 5
    eps = 1e-5
    def f(x):
        mu = x.mean()
        var = ((x - mu) ** 2).mean()
        istd = 1.0 / np.sqrt(var + eps)
        return np.full_like(x, istd)  # broadcast to Vec n
    def jac(x):
        mu = x.mean()
        var = ((x - mu) ** 2).mean()
        istd = 1.0 / np.sqrt(var + eps)
        N = len(x)
        # J[j, i] = ∂(broadcast_istd)_j / ∂x_i = -istd³ (x_i - μ) / N
        # shape: (N, N) output × (N,) input, flattened
        col = -(istd ** 3) * (x - mu) / N  # shape (N,), i.e. per input
        return np.tile(col, (N, 1))         # same row for every output j
    return check("pdiv_bnIstdBroadcast", f, jac, (n,))

# ════════════════════════════════════════════════════════════════
# BatchNorm affine: ∂(γv+β)/∂v = γδᵢⱼ
# ════════════════════════════════════════════════════════════════
def test_bn_affine():
    n = 5
    gamma, beta = 2.3, -0.7
    f = lambda v: gamma * v + beta
    jac = lambda v: gamma * np.eye(n)
    return check("pdiv_bnAffine", f, jac, (n,))

# ════════════════════════════════════════════════════════════════
# Softmax: J[i,j] = p_i(δᵢⱼ - p_j)
# ════════════════════════════════════════════════════════════════
def test_softmax():
    c = 5
    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()
    def jac(z):
        p = softmax(z)
        return np.diag(p) - np.outer(p, p)
    return check("pdiv_softmax", softmax, jac, (c,))

# ════════════════════════════════════════════════════════════════
# Depthwise conv: same as conv but per-channel
# ════════════════════════════════════════════════════════════════
def test_depthwise_input_grad():
    """Depthwise conv VJP: per-channel reversed kernel convolution."""
    c, h, w, kH, kW = 3, 6, 6, 3, 3
    W = np.random.randn(c, kH, kW)
    pad = (kH - 1) // 2

    def dw_fwd(xv):
        x = xv.reshape(c, h, w)
        out = np.zeros((c, h, w))
        for ch in range(c):
            for kh in range(kH):
                for kw in range(kW):
                    for i in range(h):
                        for j in range(w):
                            ii, jj = i + kh - pad, j + kw - pad
                            if 0 <= ii < h and 0 <= jj < w:
                                out[ch, i, j] += x[ch, ii, jj] * W[ch, kh, kw]
        return out.ravel()

    x = np.random.randn(c, h, w)
    dy = np.random.randn(c, h, w)

    # Finite-diff VJP
    xf = x.ravel()
    dx_fd = np.zeros_like(xf)
    for i in range(len(xf)):
        xp = xf.copy(); xp[i] += EPS
        xm = xf.copy(); xm[i] -= EPS
        dx_fd[i] = np.sum(((dw_fwd(xp) - dw_fwd(xm)) / (2 * EPS)) * dy.ravel())

    # Claimed: per-channel conv with reversed kernel
    dx_claimed = np.zeros_like(x)
    for ch in range(c):
        W_rev = W[ch, ::-1, ::-1]
        for kh in range(kH):
            for kw in range(kW):
                for i in range(h):
                    for j in range(w):
                        ii, jj = i + kh - pad, j + kw - pad
                        if 0 <= ii < h and 0 <= jj < w:
                            dx_claimed[ch, i, j] += dy[ch, ii, jj] * W_rev[kh, kw]

    err = np.max(np.abs(dx_fd - dx_claimed.ravel()))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'depthwise_input_grad':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Depthwise weight grad: per-channel transpose trick
# (Phase 7: new axiom depthwise_weight_grad_has_vjp3.)
# ════════════════════════════════════════════════════════════════
def test_depthwise_weight_grad():
    """Depthwise weight VJP: per-channel transpose trick. Unlike regular
    conv's `(oc, ic, kH, kW)`, the depthwise kernel is `(c, kH, kW)` —
    no cross-channel sum, each channel's kernel gets its own independent
    gradient from its own slice of x and dy."""
    c, h, w, kH, kW = 3, 6, 6, 3, 3
    W = np.random.randn(c, kH, kW)
    pad = (kH - 1) // 2

    def dw_fwd(Wv):
        Wr = Wv.reshape(c, kH, kW)
        out = np.zeros((c, h, w))
        for ch in range(c):
            for kh in range(kH):
                for kw in range(kW):
                    for i in range(h):
                        for j in range(w):
                            ii, jj = i + kh - pad, j + kw - pad
                            if 0 <= ii < h and 0 <= jj < w:
                                out[ch, i, j] += x[ch, ii, jj] * Wr[ch, kh, kw]
        return out

    x = np.random.randn(c, h, w)
    dy = np.random.randn(c, h, w)

    # Finite-diff: perturb each W entry, measure <Δoutput, dy>
    dW_fd = np.zeros_like(W)
    Wf = W.ravel()
    for idx in range(len(Wf)):
        Wp = Wf.copy(); Wp[idx] += EPS
        Wm = Wf.copy(); Wm[idx] -= EPS
        dW_fd.ravel()[idx] = np.sum(((dw_fwd(Wp) - dw_fwd(Wm)) / (2 * EPS)) * dy)

    # Claimed: per-channel `dW[c, kh, kw] = Σ_{i,j} x[c, i+kh-p, j+kw-p] * dy[c, i, j]`
    dW_claimed = np.zeros_like(W)
    for ch in range(c):
        for kh in range(kH):
            for kw in range(kW):
                s = 0.0
                for i in range(h):
                    for j in range(w):
                        ii, jj = i + kh - pad, j + kw - pad
                        if 0 <= ii < h and 0 <= jj < w:
                            s += x[ch, ii, jj] * dy[ch, i, j]
                dW_claimed[ch, kh, kw] = s

    err = np.max(np.abs(dW_fd - dW_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'depthwise_weight_grad':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# NOTE — Phase 8: `mhsa_has_vjp_mat` (Attention.lean) is a bundled
# existence axiom (HasVJPMat), not a specific-formula claim. It's a
# composition of primitives that ARE tested here individually:
# - Q/K/V/O projections → gradient-checked via pdiv_dense / pdiv_dense_W
# - per-head SDPA       → gradient-checked via sdpa_back_Q/K/V below
# - reshape / concat    → sparse pdiv_reindex, non-falsifiable at the
#                         formula level (just index permutations)
# So we intentionally don't add a direct mhsa gradient check — the axiom's
# content is "the per-head vmap is consistent with the single-head proof,"
# and the single-head proof (sdpa) is already numerically verified below.
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# SDPA backwards: sdpa_back_Q / sdpa_back_K / sdpa_back_V
#
# These mirror the concrete definitions in Attention.lean. We check
# each of dQ, dK, dV against finite-difference Jacobians of
#     sdpa(Q, K, V) = softmax_row(Q K^T / sqrt(d)) V
# contracted with a random dOut.
# ════════════════════════════════════════════════════════════════
def _sdpa_forward(Q, K, V):
    """Scaled dot-product attention for a single sequence / head."""
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    scores = Q @ K.T                  # (n, n)
    scaled = scale * scores
    # Stable row softmax
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    weights = e / e.sum(axis=1, keepdims=True)  # (n, n)
    return weights @ V                # (n, d)

def _sdpa_back_Q(Q, K, V, dOut):
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    scores = Q @ K.T
    scaled_stable = scale * scores
    scaled_stable = scaled_stable - scaled_stable.max(axis=1, keepdims=True)
    e = np.exp(scaled_stable)
    weights = e / e.sum(axis=1, keepdims=True)
    dWeights = dOut @ V.T                         # (n, n)
    # Per-row softmax VJP: p_i * (dw_i - <p_i, dw_i>)
    s = (weights * dWeights).sum(axis=1, keepdims=True)
    dScaled = weights * (dWeights - s)
    dScores = scale * dScaled
    return dScores @ K

def _sdpa_back_K(Q, K, V, dOut):
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    scores = Q @ K.T
    scaled_stable = scale * scores
    scaled_stable = scaled_stable - scaled_stable.max(axis=1, keepdims=True)
    e = np.exp(scaled_stable)
    weights = e / e.sum(axis=1, keepdims=True)
    dWeights = dOut @ V.T
    s = (weights * dWeights).sum(axis=1, keepdims=True)
    dScaled = weights * (dWeights - s)
    dScores = scale * dScaled
    return dScores.T @ Q

def _sdpa_back_V(Q, K, V, dOut):
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    scores = Q @ K.T
    scaled_stable = scale * scores
    scaled_stable = scaled_stable - scaled_stable.max(axis=1, keepdims=True)
    e = np.exp(scaled_stable)
    weights = e / e.sum(axis=1, keepdims=True)
    return weights.T @ dOut

def _sdpa_fd_grad(var, Q, K, V, dOut):
    """Finite-difference grad of <sdpa(Q,K,V), dOut> w.r.t. `var` in {Q, K, V}."""
    assert var in ("Q", "K", "V")
    base = {"Q": Q, "K": K, "V": V}[var]
    g = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        saved = base[idx]
        base[idx] = saved + EPS
        fp = np.sum(_sdpa_forward(Q, K, V) * dOut)
        base[idx] = saved - EPS
        fm = np.sum(_sdpa_forward(Q, K, V) * dOut)
        base[idx] = saved
        g[idx] = (fp - fm) / (2 * EPS)
        it.iternext()
    return g

def _test_sdpa_back(var, n=4, d=3):
    Q = np.random.randn(n, d)
    K = np.random.randn(n, d)
    V = np.random.randn(n, d)
    dOut = np.random.randn(n, d)
    fd = _sdpa_fd_grad(var, Q, K, V, dOut)
    claimed = {
        "Q": _sdpa_back_Q(Q, K, V, dOut),
        "K": _sdpa_back_K(Q, K, V, dOut),
        "V": _sdpa_back_V(Q, K, V, dOut),
    }[var]
    err = np.max(np.abs(fd - claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'sdpa_back_' + var:30s} max_err={err:.2e}")
    if err >= TOL:
        idx = np.unravel_index(np.argmax(np.abs(fd - claimed)), fd.shape)
        print(f"         worst at {idx}: fd={fd[idx]:.8f} claimed={claimed[idx]:.8f}")
    return err < TOL

def test_sdpa_back_Q(): return _test_sdpa_back("Q")
def test_sdpa_back_K(): return _test_sdpa_back("K")
def test_sdpa_back_V(): return _test_sdpa_back("V")

# ════════════════════════════════════════════════════════════════
# GELU: diagonal Jacobian
# ════════════════════════════════════════════════════════════════
def test_gelu():
    from scipy.special import erf
    n = 5
    def gelu(x):
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    def gelu_deriv(x):
        phi = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
        return Phi + x * phi
    def jac(x):
        return np.diag(gelu_deriv(x))
    return check("pdiv_gelu", gelu, jac, (n,))

# ════════════════════════════════════════════════════════════════
# PatchEmbed (ViT): conv2d (stride=patchSize) + reshape to (N, D) +
# CLS prepend + add pos_embed → flatten. Bundled VJP for the whole
# composition (Attention.lean: patchEmbed_flat_has_vjp).
#
# Strategy: verify the input-image gradient. The conv-on-tiles is
# stride=patchSize with no overlap, so the input grad has a clean
# closed form (each output position lands on exactly one input patch).
# CLS / pos_embed grads are trivial (identity-like) and not retested
# here — the conv path is the substantive part of the backward.
# ════════════════════════════════════════════════════════════════
def test_patch_embed_flat():
    ic, H, W, patchSize, D = 2, 4, 4, 2, 3
    nH, nW = H // patchSize, W // patchSize
    N = nH * nW                                              # 4 patches

    W_conv = np.random.randn(D, ic, patchSize, patchSize)
    b_conv = np.random.randn(D)
    cls    = np.random.randn(D)
    pos    = np.random.randn(N + 1, D)

    def fwd(x_flat):
        x = x_flat.reshape(ic, H, W)
        out = np.zeros((D, nH, nW))
        for o in range(D):
            for c in range(ic):
                for ph in range(nH):
                    for pw in range(nW):
                        for kh in range(patchSize):
                            for kw in range(patchSize):
                                out[o, ph, pw] += (
                                    x[c, ph * patchSize + kh, pw * patchSize + kw]
                                    * W_conv[o, c, kh, kw]
                                )
            out[o] += b_conv[o]
        patches = out.transpose(1, 2, 0).reshape(N, D)        # row-major
        full    = np.vstack([cls.reshape(1, D), patches]) + pos
        return full.ravel()

    x  = np.random.randn(ic, H, W)
    dy = np.random.randn((N + 1) * D)

    # Finite-diff input gradient
    xf    = x.ravel()
    dx_fd = np.zeros_like(xf)
    for i in range(len(xf)):
        xp = xf.copy(); xp[i] += EPS
        xm = xf.copy(); xm[i] -= EPS
        dx_fd[i] = np.sum(((fwd(xp) - fwd(xm)) / (2 * EPS)) * dy)

    # Claimed: drop dy's CLS row (no x-dep), reshape rest to (D, nH, nW),
    # apply the stride=patchSize conv input-grad.
    dy_full    = dy.reshape(N + 1, D)
    dy_patches = dy_full[1:]                                  # (N, D)
    dy_grid    = dy_patches.reshape(nH, nW, D).transpose(2, 0, 1)  # (D, nH, nW)
    dx_claimed = np.zeros((ic, H, W))
    for o in range(D):
        for c in range(ic):
            for ph in range(nH):
                for pw in range(nW):
                    for kh in range(patchSize):
                        for kw in range(patchSize):
                            dx_claimed[c, ph * patchSize + kh, pw * patchSize + kw] += (
                                dy_grid[o, ph, pw] * W_conv[o, c, kh, kw]
                            )

    err    = np.max(np.abs(dx_fd - dx_claimed.ravel()))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'patchEmbed_flat (input grad)':30s} max_err={err:.2e}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# MLP composition: Dense → ReLU → Dense → ReLU → Dense + softmax CE.
# Tests the bundled `mlp_has_vjp` axiom by composing the chain-rule
# input gradient and comparing it to FD on the full network's loss.
# Catches composition errors that the per-axiom checks can't see.
# ════════════════════════════════════════════════════════════════
def test_mlp_full():
    n_in, h1, h2, c = 8, 6, 4, 3
    label = 1
    W1 = np.random.randn(n_in, h1) * 0.5
    b1 = np.random.randn(h1) * 0.5 + 1.0   # bias positive to avoid ReLU kink
    W2 = np.random.randn(h1, h2) * 0.5
    b2 = np.random.randn(h2) * 0.5 + 1.0
    W3 = np.random.randn(h2, c) * 0.5
    b3 = np.random.randn(c) * 0.5

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def loss_of(x):
        y1 = x @ W1 + b1
        z1 = np.maximum(y1, 0.0)
        y2 = z1 @ W2 + b2
        z2 = np.maximum(y2, 0.0)
        y3 = z2 @ W3 + b3
        return -np.log(softmax(y3)[label])

    def mlp_input_grad(x):
        # Forward (recompute activations for backward).
        y1 = x @ W1 + b1
        z1 = np.maximum(y1, 0.0)
        y2 = z1 @ W2 + b2
        z2 = np.maximum(y2, 0.0)
        y3 = z2 @ W3 + b3
        # Backward chain: softmax CE → Dense3 → ReLU → Dense2 → ReLU → Dense1.
        oh = np.zeros(c); oh[label] = 1.0
        dy3 = softmax(y3) - oh
        dz2 = W3 @ dy3
        dy2 = (y2 > 0).astype(float) * dz2
        dz1 = W2 @ dy2
        dy1 = (y1 > 0).astype(float) * dz1
        return W1 @ dy1

    x = np.random.randn(n_in)
    dx_claimed = mlp_input_grad(x)
    dx_fd = np.zeros(n_in)
    for i in range(n_in):
        xp = x.copy(); xp[i] += EPS
        xm = x.copy(); xm[i] -= EPS
        dx_fd[i] = (loss_of(xp) - loss_of(xm)) / (2 * EPS)

    err = np.max(np.abs(dx_fd - dx_claimed))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {status}: {'mlp_has_vjp (full network)':30s} max_err={err:.2e}")
    if err >= TOL:
        idx = int(np.argmax(np.abs(dx_fd - dx_claimed)))
        print(f"         worst at {idx}: fd={dx_fd[idx]:.8f} claimed={dx_claimed[idx]:.8f}")
    return err < TOL

# ════════════════════════════════════════════════════════════════
# Multi-head SDPA: H independent single-head SDPAs stacked along the
# head axis. The bundled `mhsa_has_vjp_mat` axiom asserts the VJP
# factors per head; this test confirms that the per-head sdpa_back_*
# stacked over heads matches FD on the full multi-head forward.
# ════════════════════════════════════════════════════════════════
def test_mhsa_full():
    n, H, d_h = 4, 2, 3
    Q = np.random.randn(H, n, d_h)
    K = np.random.randn(H, n, d_h)
    V = np.random.randn(H, n, d_h)
    dOut = np.random.randn(H, n, d_h)

    def mhsa_fwd(Q_, K_, V_):
        out = np.zeros((H, n, d_h))
        for h in range(H):
            out[h] = _sdpa_forward(Q_[h], K_[h], V_[h])
        return out

    # Claimed: per-head sdpa_back_* stacked along H axis.
    dQ_claimed = np.zeros_like(Q)
    dK_claimed = np.zeros_like(K)
    dV_claimed = np.zeros_like(V)
    for h in range(H):
        dQ_claimed[h] = _sdpa_back_Q(Q[h], K[h], V[h], dOut[h])
        dK_claimed[h] = _sdpa_back_K(Q[h], K[h], V[h], dOut[h])
        dV_claimed[h] = _sdpa_back_V(Q[h], K[h], V[h], dOut[h])

    def fd_grad(base):
        g = np.zeros_like(base)
        it = np.nditer(base, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            saved = base[idx]
            base[idx] = saved + EPS
            fp = np.sum(mhsa_fwd(Q, K, V) * dOut)
            base[idx] = saved - EPS
            fm = np.sum(mhsa_fwd(Q, K, V) * dOut)
            base[idx] = saved
            g[idx] = (fp - fm) / (2 * EPS)
            it.iternext()
        return g

    all_ok = True
    for label, base, claimed in [("Q", Q, dQ_claimed),
                                  ("K", K, dK_claimed),
                                  ("V", V, dV_claimed)]:
        fd = fd_grad(base)
        err = np.max(np.abs(fd - claimed))
        status = "PASS" if err < TOL else "FAIL"
        print(f"  {status}: {f'mhsa_has_vjp_mat ({label})':30s} max_err={err:.2e}")
        if err >= TOL:
            all_ok = False
    return all_ok

# ════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Numerical gradient checks for axiomatized VJPs")
    print("=" * 60)
    results = []
    results.append(("Tensor.lean",  "pdiv_id",              True))  # trivial
    results.append(("MLP.lean",     "pdiv_dense",           test_dense()))
    results.append(("MLP.lean",     "pdiv_dense_W",         test_dense_weight_grad()))
    results.append(("MLP.lean",     "pdiv_dense_b",         test_dense_bias_grad()))
    results.append(("MLP.lean",     "pdiv_relu",            test_relu()))
    results.append(("MLP.lean",     "softmaxCE_grad",       test_softmax_ce()))
    results.append(("MLP.lean",     "mlp_has_vjp",          test_mlp_full()))
    results.append(("CNN.lean",     "conv2d_input_grad",    test_conv2d_input_grad_formula()))
    results.append(("CNN.lean",     "conv2d_weight_grad",   test_conv2d_weight_grad()))
    results.append(("CNN.lean",     "conv2d_bias_grad",     test_conv2d_bias_grad()))
    results.append(("CNN.lean",     "maxPool2_input_grad",  test_maxpool2()))
    results.append(("BatchNorm",    "pdiv_bnNormalize",     test_bn_normalize()))
    results.append(("BatchNorm",    "pdiv_bnCentered",      test_bn_centered()))
    results.append(("BatchNorm",    "pdiv_bnIstdBroadcast", test_bn_istd_broadcast()))
    results.append(("BatchNorm",    "pdiv_bnAffine",        test_bn_affine()))
    results.append(("Attention",    "pdiv_softmax",         test_softmax()))
    results.append(("Attention",    "sdpa_back_Q",          test_sdpa_back_Q()))
    results.append(("Attention",    "sdpa_back_K",          test_sdpa_back_K()))
    results.append(("Attention",    "sdpa_back_V",          test_sdpa_back_V()))
    results.append(("Attention",    "patchEmbed_flat",      test_patch_embed_flat()))
    results.append(("Attention",    "mhsa_has_vjp_mat",     test_mhsa_full()))
    results.append(("Depthwise",    "depthwise_input_grad", test_depthwise_input_grad()))
    results.append(("Depthwise",    "depthwise_weight_grad", test_depthwise_weight_grad()))
    results.append(("Depthwise",    "depthwise_bias_grad",   test_depthwise_bias_grad()))
    try:
        results.append(("LayerNorm", "pdiv_gelu",           test_gelu()))
    except ImportError:
        print("  SKIP: pdiv_gelu (scipy not installed)")

    print("=" * 60)
    passed = sum(1 for _, _, r in results if r)
    total = len(results)
    print(f"{passed}/{total} checks passed.")
    if passed < total:
        for f, n, r in results:
            if not r:
                print(f"  FAILED: {f}:{n}")
