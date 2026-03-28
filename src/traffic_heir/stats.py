"""
stats.py
--------
Statistical utilities for rigorous evaluation in the Information Fusion paper.
Pure-Python implementation (no scipy/numpy dependency) for consistency with
the rest of the prototype.

Functions
---------
bootstrap_ci      : Bootstrap confidence interval for classifier accuracy.
paired_ttest      : Paired t-test between two lists of per-seed accuracies.
mcnemar_test      : McNemar's test comparing two classifiers on the same test set.
effect_size_cohens_d : Cohen's d effect size between two accuracy distributions.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    n_boot: int = 1000,
    alpha: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for accuracy.

    Parameters
    ----------
    y_true, y_pred : ground-truth and predicted labels (same length)
    n_boot         : number of bootstrap resamples
    alpha          : confidence level (default 0.95 → 95% CI)
    seed           : RNG seed for reproducibility

    Returns
    -------
    (ci_lower, ci_upper, mean_acc)
    """
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0, 0.0)

    rng = random.Random(seed)
    correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    base_acc = sum(correct) / n

    boot_accs: List[float] = []
    for _ in range(n_boot):
        sample = [correct[rng.randrange(n)] for _ in range(n)]
        boot_accs.append(sum(sample) / n)

    boot_accs.sort()
    tail = (1.0 - alpha) / 2.0
    lo_idx = max(0, int(math.floor(tail * n_boot)))
    hi_idx = min(n_boot - 1, int(math.ceil((1.0 - tail) * n_boot)) - 1)
    return (boot_accs[lo_idx], boot_accs[hi_idx], base_acc)


# ---------------------------------------------------------------------------
# Paired t-test (pure Python)
# ---------------------------------------------------------------------------

def paired_ttest(
    accs_a: List[float],
    accs_b: List[float],
) -> Tuple[float, float]:
    """
    Paired t-test between two lists of per-seed accuracies.

    Parameters
    ----------
    accs_a, accs_b : per-seed accuracies for model A and model B

    Returns
    -------
    (t_stat, p_value)
        p_value is two-tailed, approximated via the t-distribution CDF.
    """
    n = len(accs_a)
    if n != len(accs_b) or n < 2:
        return (0.0, 1.0)

    diffs = [a - b for a, b in zip(accs_a, accs_b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n) if var_d > 0 else 1e-12
    t_stat = mean_d / se
    df = n - 1
    p_value = _t_pvalue(t_stat, df)
    return (t_stat, p_value)


def _t_pvalue(t: float, df: int) -> float:
    """
    Two-tailed p-value from t-distribution using a numerical approximation.
    Accurate to ~3 decimal places for df >= 2.
    Uses the regularised incomplete beta function approximation.
    """
    x = df / (df + t * t)
    p_one_tail = 0.5 * _regularised_incomplete_beta(x, df / 2.0, 0.5)
    return min(1.0, 2.0 * p_one_tail)


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """
    Regularised incomplete beta function I_x(a, b) via continued fraction.
    Sufficient precision for hypothesis testing with small df.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use symmetry relation when x > (a+1)/(a+b+2)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularised_incomplete_beta(1.0 - x, b, a)

    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    # Lentz continued fraction
    f, c, d = 1.0, 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / d if abs(d) > 1e-30 else 1e30
    f = d
    for m in range(1, 200):
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / d if abs(d) > 1e-30 else 1e30
        c = c if abs(c) > 1e-30 else 1e30
        delta = c * d
        f *= delta
        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / d if abs(d) > 1e-30 else 1e30
        c = c if abs(c) > 1e-30 else 1e30
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * f


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: List[int],
    preds_a: List[int],
    preds_b: List[int],
) -> Tuple[float, float]:
    """
    McNemar's test comparing two classifiers on the same test set.

    b = count where A correct, B wrong
    c = count where A wrong,    B correct
    chi2 = (|b - c| - 1)^2 / (b + c)   (with continuity correction)

    Returns
    -------
    (chi2_stat, p_value)  — p_value from chi-squared distribution with df=1
    """
    b = sum(1 for t, a, bb in zip(y_true, preds_a, preds_b) if t == a and t != bb)
    c = sum(1 for t, a, bb in zip(y_true, preds_a, preds_b) if t != a and t == bb)
    if b + c == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = _chi2_pvalue(chi2, df=1)
    return (chi2, p_value)


def _chi2_pvalue(chi2: float, df: int = 1) -> float:
    """P-value from chi-squared distribution (df=1 only, sufficient here)."""
    if chi2 <= 0:
        return 1.0
    # For df=1: chi2 ~ Normal(0,1)^2, so p = 2 * (1 - Phi(sqrt(chi2)))
    z = math.sqrt(chi2)
    return 2.0 * (1.0 - _standard_normal_cdf(z))


def _standard_normal_cdf(x: float) -> float:
    """CDF of standard normal using math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------

def effect_size_cohens_d(
    accs_a: List[float],
    accs_b: List[float],
) -> float:
    """
    Cohen's d = (mean_a - mean_b) / pooled_std.
    Positive d means model A is better.
    Interpretation: |d| < 0.2 small, 0.2–0.5 medium, > 0.5 large.
    """
    n_a, n_b = len(accs_a), len(accs_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = sum(accs_a) / n_a
    mean_b = sum(accs_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in accs_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in accs_b) / (n_b - 1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (mean_a - mean_b) / pooled_std
