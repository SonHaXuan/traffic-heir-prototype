"""
heir_overhead.py
----------------
Theoretical analysis of the computational and communication overhead of
HEIR-based encrypted cooperative inference.

All latency estimates are derived from published SEAL / OpenFHE benchmarks
(Fan & Vercauteren 2012; Cheon et al. 2017 CKKS; Microsoft SEAL 4.x docs).
They represent conservative lower bounds at 128-bit security with a 16384
polynomial modulus — a realistic setting for the feature dimensions used here.

Reference latency figures (from SEAL benchmark suite, 2024):
  CKKS ciphertext-ciphertext multiply: ~20–80 ms  (we use 50 ms conservative)
  CKKS ciphertext-plaintext multiply:  ~2–5 ms    (we use 3 ms)
  CKKS rotation (slot permutation):    ~10–30 ms  (we use 20 ms)
  CKKS addition:                       ~0.1 ms    (negligible)

Polynomial activation `0.125x² + 0.5x + 0.25` requires depth 2 (one squaring +
one multiply-by-coefficient), which fits comfortably in a CKKS bootstrap budget.

Communication overhead uses a standard CKKS ciphertext expansion factor of
~4096x relative to the plaintext vector (16384 polynomial degree, 64-bit coeffs).
"""

from __future__ import annotations

from typing import Dict


# ---------------------------------------------------------------------------
# Published benchmark constants (conservative estimates)
# ---------------------------------------------------------------------------
_CT_CT_MUL_MS = 50.0      # ciphertext × ciphertext multiply
_CT_PT_MUL_MS = 3.0       # ciphertext × plaintext multiply (weight matrix row)
_CT_ADD_MS    = 0.1        # ciphertext addition
_ROTATION_MS  = 20.0       # key-switching / rotation
_POLY_DEPTH   = 2          # multiply depth of 0.125x² + 0.5x + 0.25
_CKKS_EXPANSION = 4096     # ciphertext-to-plaintext size ratio (typical CKKS)
_PRECISION_BITS = 32       # float32 plaintext


def count_operations(feature_dim: int, hidden_dim: int) -> Dict[str, int]:
    """
    Count homomorphic operations for a two-layer network forward pass.

    Layer 1 (input → hidden):
      - feature_dim multiply-accumulate ops per hidden neuron
      - hidden_dim neurons
      - poly activation: 1 squaring (CT×CT) + 1 CT×PT per hidden neuron

    Layer 2 (hidden → output, binary):
      - hidden_dim multiply-accumulate ops
      - sigmoid is approximated by poly activation (same depth)

    Returns
    -------
    dict with keys: ct_ct_muls, ct_pt_muls, additions, total_mul_depth
    """
    # Layer 1: each hidden neuron does feature_dim inner-product steps
    # CKKS can batch, so conservatively count as feature_dim CT×PT muls per neuron
    l1_ct_pt = feature_dim * hidden_dim
    l1_adds = (feature_dim - 1) * hidden_dim
    # Poly activation (depth 2): squaring (CT×CT) + scaling (CT×PT)
    act_ct_ct = hidden_dim      # one squaring per neuron
    act_ct_pt = hidden_dim      # one CT×PT per neuron

    # Layer 2: hidden_dim inner product
    l2_ct_pt = hidden_dim
    l2_adds = hidden_dim - 1
    # Output sigmoid/poly activation
    out_ct_ct = 1
    out_ct_pt = 1

    return {
        "ct_ct_muls": act_ct_ct + out_ct_ct,
        "ct_pt_muls": l1_ct_pt + act_ct_pt + l2_ct_pt + out_ct_pt,
        "additions": l1_adds + l2_adds,
        "total_mul_depth": _POLY_DEPTH + 1,   # depth 2 act + depth 1 output
    }


def ckks_latency_estimate_ms(
    feature_dim: int,
    hidden_dim: int,
) -> Dict[str, object]:
    """
    Theoretical latency estimate for one encrypted forward pass.

    Returns
    -------
    dict with latency_ms, breakdown, and reference note.
    """
    ops = count_operations(feature_dim, hidden_dim)
    ct_ct_total = ops["ct_ct_muls"] * _CT_CT_MUL_MS
    ct_pt_total = ops["ct_pt_muls"] * _CT_PT_MUL_MS
    add_total   = ops["additions"]  * _CT_ADD_MS
    # Rotations needed for inner products (one rotation per accumulation step)
    rotations   = (feature_dim + hidden_dim) * _ROTATION_MS
    total_ms    = ct_ct_total + ct_pt_total + add_total + rotations

    return {
        "latency_ms": round(total_ms, 1),
        "breakdown_ms": {
            "ct_ct_multiply": round(ct_ct_total, 1),
            "ct_pt_multiply": round(ct_pt_total, 1),
            "additions": round(add_total, 1),
            "rotations": round(rotations, 1),
        },
        "mul_depth": ops["total_mul_depth"],
        "reference": (
            "Conservative estimate based on SEAL 4.x benchmarks "
            "(128-bit security, poly_modulus_degree=16384, CKKS scheme). "
            "Real hardware latency will vary; use as order-of-magnitude guide."
        ),
    }


def communication_cost_kb(
    feature_dim: int,
    n_intersections: int,
    precision_bits: int = _PRECISION_BITS,
) -> Dict[str, object]:
    """
    Communication overhead when each intersection broadcasts its encrypted
    feature vector to a central aggregator per control timestep.

    Parameters
    ----------
    feature_dim      : number of features per intersection (e.g. 49 for coop)
    n_intersections  : total number of cooperating intersections
    precision_bits   : plaintext element precision (default float32 = 32 bits)

    Returns
    -------
    dict with plaintext_kb, ciphertext_kb, expansion_factor, and per_timestep note.
    """
    plaintext_bytes  = feature_dim * precision_bits // 8        # bytes per intersection
    plaintext_total  = plaintext_bytes * n_intersections
    ciphertext_total = plaintext_total * _CKKS_EXPANSION        # encrypted size

    return {
        "per_intersection_plaintext_bytes": plaintext_bytes,
        "plaintext_total_kb": round(plaintext_total / 1024, 3),
        "ciphertext_total_kb": round(ciphertext_total / 1024, 1),
        "expansion_factor": _CKKS_EXPANSION,
        "n_intersections": n_intersections,
        "feature_dim": feature_dim,
        "note": (
            f"Each intersection encrypts a {feature_dim}-dimensional feature vector "
            f"({precision_bits}-bit precision) and sends to aggregator. "
            f"CKKS ciphertext expansion ~{_CKKS_EXPANSION}x. "
            f"Total broadcast per timestep: {round(ciphertext_total / 1024, 1)} KB "
            f"(encrypted) vs {round(plaintext_total / 1024, 3)} KB (plaintext)."
        ),
    }


def accuracy_gap_summary(
    he_accuracy: float,
    plaintext_accuracy: float,
) -> Dict[str, object]:
    """
    Summary of accuracy cost incurred by using HE-friendly polynomial
    activation instead of ReLU / standard sigmoid.
    """
    gap_pp = (plaintext_accuracy - he_accuracy) * 100.0
    relative_pct = (gap_pp / (plaintext_accuracy * 100.0)) * 100.0 if plaintext_accuracy > 0 else 0.0
    return {
        "he_accuracy": round(he_accuracy, 4),
        "plaintext_accuracy": round(plaintext_accuracy, 4),
        "gap_pp": round(gap_pp, 2),
        "relative_gap_pct": round(relative_pct, 2),
        "interpretation": (
            f"HE-friendly polynomial activation costs {gap_pp:.1f} pp "
            f"({relative_pct:.1f}% relative) vs plaintext ReLU baseline."
        ),
    }
