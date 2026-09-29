"""
prime_residue_spectral.py

Spectral analysis of Markov chains on prime residues, with deep
gap exploration, self-verification, and figure generation.

Analysis pipeline:
  1. Residue analysis modulo 210, 2310, and 30030.
  2. Gap analysis: real gaps vs shuffle, classical Cramer, and
     refined Cramer (density + observed gap distribution).
  3. Block-wise gap analysis with binned gaps.
  4. Significance test of gaps against shuffle.
  5. Normalized gaps vs refined Cramer normalized by local density.
  6. Residue-gap correlation, normalized by local density.
  7. Stationarity of gaps (first half vs second half).
  8. KS test of normalized gaps against exponential.
  9. Extended exploration:
       - Larger prime sample.
       - Gap analysis across 10 narrow ranges, with normalization
         applied within each range.
       - Sliding-window density normalization with trend test.
 10. Self-verification of all computed quantities.
 11. Generation of six publication-quality figures (PNG + PDF).

Usage:
    python3 prime_residue_spectral.py
"""

import numpy as np
from sympy import primerange, isprime
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time


# ============================================================
# CONFIGURATION
# ============================================================
N_PRIMOS = 500_000
MODULOS = [210, 2310, 30030]
SEMILLA = 42
ALPHA = 0.5
N_SHUFFLES_NULOS = 50
N_SHUFFLES_GAPS = 100
BLOCK_SIZE = 2000
N_BINS_GAPS_BLOCKS = 10
N_BINS_GAPS_KS = 20
N_WINDOWS_DENSITY = 100
VERBOSE_SELFCHECK = False

# 10 narrow ranges for gap analysis
NARROW_RANGES = [
    (10_000, 100_000),
    (100_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 5_000_000),
    (5_000_000, 10_000_000),
    (10_000_000, 50_000_000),
    (50_000_000, 100_000_000),
    (100_000_000, 200_000_000),
    (200_000_000, 300_000_000),
    (300_000_000, 500_000_000),
]

# Minimum primes required in a narrow range to attempt analysis
MIN_PRIMES_RANGE = 500


# ============================================================
# SELF-VERIFIER
# ============================================================
class SelfVerifier:
    """Accumulates verification checks and reports at the end."""
    def __init__(self):
        self.checks = []

    def check(self, name, condition, detail=""):
        self.checks.append({
            'name': name,
            'ok': bool(condition),
            'detail': detail,
        })
        if VERBOSE_SELFCHECK:
            symbol = "OK " if condition else "FAIL"
            print(f"    [{symbol}] {name}: {detail}")

    def summary(self):
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c['ok'])
        print("\n" + "=" * 80)
        print("  SELF-VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"  Passed: {passed}/{total}")
        if passed < total:
            print("\n  FAILED CHECKS:")
            for c in self.checks:
                if not c['ok']:
                    print(f"    [FAIL] {c['name']}: {c['detail']}")
        else:
            print("  All checks passed.")
        return passed == total


# ============================================================
# UTILITIES
# ============================================================
def _phi(n):
    """Euler totient function."""
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result


def bin_values(values, n_bins, bins=None):
    """Bin continuous values into n_bins by percentiles."""
    if bins is None:
        bins = np.percentile(values, np.linspace(0, 100, n_bins + 1))
        bins[0] -= 1e-10
        bins[-1] += 1e-10
    obs = np.digitize(values, bins) - 1
    return np.clip(obs, 0, n_bins - 1), bins


# ============================================================
# DATA GENERATION
# ============================================================
def generate_primes(n, verifier=None):
    """Generate the first n primes."""
    print(f"Generating {n:,} primes...")
    limit = int(n * (np.log(n) + np.log(np.log(n)))) + 10
    primes = list(primerange(2, limit))
    while len(primes) < n:
        limit *= 2
        primes = list(primerange(2, limit))
    primes = primes[:n]

    if verifier is not None:
        all_prime = all(isprime(p) for p in primes[:1000])
        verifier.check("Generated numbers are prime", all_prime,
                       f"verified first 1000 of {n}")
        sorted_ok = all(primes[i] < primes[i+1]
                        for i in range(min(len(primes)-1, 1000)))
        verifier.check("Primes in increasing order", sorted_ok,
                       "verified first 1000")
        verifier.check("Correct number of primes", len(primes) == n,
                       f"len={len(primes)}, expected={n}")

    return primes


def residues_to_indices(primes, modulo, verifier=None):
    """Convert primes to indices of residues coprime to modulo."""
    coprimes = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    mapping = {r: i for i, r in enumerate(coprimes)}
    obs = np.array([mapping[p % modulo] for p in primes
                    if np.gcd(p, modulo) == 1])

    if verifier is not None:
        in_range = np.all((obs >= 0) & (obs < len(coprimes)))
        verifier.check(f"Indices mod {modulo} in range", in_range,
                       f"range=[0, {len(coprimes)-1}], "
                       f"min={obs.min()}, max={obs.max()}")
        verifier.check(f"Number of coprime residues mod {modulo}",
                       len(coprimes) == _phi(modulo),
                       f"len={len(coprimes)}, phi({modulo})={_phi(modulo)}")

    return obs, coprimes


def compute_gaps(primes, verifier=None):
    """Compute gaps between consecutive primes."""
    primes_arr = np.array(primes)
    gaps = np.diff(primes_arr)

    if verifier is not None:
        verifier.check("All gaps positive", np.all(gaps > 0),
                       f"min={gaps.min()}, max={gaps.max()}")
        even_ok = np.all(gaps[1:] % 2 == 0)
        verifier.check("Even gaps (except first)", even_ok,
                       f"gaps[0]={gaps[0]}, rest even")
        sum_ok = (gaps.sum() == primes_arr[-1] - primes_arr[0])
        verifier.check("Sum of gaps = last - first prime", sum_ok,
                       f"sum={gaps.sum()}, "
                       f"diff={primes_arr[-1]-primes_arr[0]}")

    return gaps


def generate_cramer_classical(n, modulo, seed, verifier=None):
    """Classical Cramer null: density 1/log x."""
    rng = np.random.default_rng(seed)
    coprimes = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    mapping = {r: i for i, r in enumerate(coprimes)}
    result = []
    x = 7
    while len(result) < n:
        p = (modulo / len(coprimes)) / np.log(x)
        if rng.random() < p and np.gcd(x, modulo) == 1:
            result.append(mapping[x % modulo])
        x += 1
    return np.array(result)


def generate_cramer_refined(n, modulo, observed_gaps, seed):
    """Refined Cramer: density + observed gap distribution."""
    rng = np.random.default_rng(seed)
    coprimes = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    mapping = {r: i for i, r in enumerate(coprimes)}

    gap_values, counts = np.unique(observed_gaps, return_counts=True)
    probs = counts / counts.sum()

    result = []
    p_current = 7
    while len(result) < n:
        gap = rng.choice(gap_values, p=probs)
        p_current += int(gap)
        while np.gcd(p_current, modulo) != 1:
            p_current += 1
        result.append(mapping[p_current % modulo])

    return np.array(result)


def generate_pseudo_primes_with_gaps(n, observed_gaps, seed):
    """Generate pseudo-primes with the observed gap distribution."""
    rng = np.random.default_rng(seed)
    gap_values, counts = np.unique(observed_gaps, return_counts=True)
    probs = counts / counts.sum()

    p_current = 7
    primes = [p_current]
    while len(primes) < n:
        gap = rng.choice(gap_values, p=probs)
        p_current += int(gap)
        primes.append(p_current)
    return np.array(primes)


def shuffle_observations(obs, seed):
    """Shuffle observed sequence."""
    rng = np.random.default_rng(seed)
    obs_shuffled = obs.copy()
    rng.shuffle(obs_shuffled)
    return obs_shuffled


# ============================================================
# TRANSITION MATRIX
# ============================================================
def transition_matrix_order1(obs, K, alpha=ALPHA, verifier=None,
                             name="P"):
    """Order-1 transition matrix with Dirichlet smoothing."""
    counts = np.zeros((K, K))
    for t in range(1, len(obs)):
        counts[obs[t - 1], obs[t]] += 1

    P = np.zeros((K, K))
    for i in range(K):
        den = counts[i].sum()
        P[i] = (counts[i] + alpha) / (den + alpha * K)

    if verifier is not None:
        row_sums = P.sum(axis=1)
        rows_ok = np.allclose(row_sums, 1.0, atol=1e-10)
        verifier.check(f"{name}: rows sum to 1", rows_ok,
                       f"min={row_sums.min():.12f}, "
                       f"max={row_sums.max():.12f}")
        verifier.check(f"{name}: non-negative entries", np.all(P >= 0),
                       f"min={P.min():.2e}")
        verifier.check(f"{name}: positive entries (smoothing)",
                       np.all(P > 0), f"min={P.min():.2e}")
        total_counts = counts.sum()
        verifier.check(f"{name}: sum of counts",
                       total_counts == len(obs) - 1,
                       f"sum={total_counts}, expected={len(obs)-1}")

    return P, counts


def transition_matrix_order2(obs, K, alpha=ALPHA):
    """Order-2 transition matrix."""
    context_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    T = len(obs)
    for t in range(2, T):
        context = (obs[t - 2], obs[t - 1])
        nxt = obs[t]
        context_counts[context] += 1
        transition_counts[(context, nxt)] += 1

    P2 = {}
    for ctx in context_counts:
        den = context_counts[ctx]
        vec = np.zeros(K)
        for k in range(K):
            num = transition_counts.get((ctx, k), 0)
            vec[k] = (num + alpha) / (den + alpha * K)
        P2[ctx] = vec
    return P2, context_counts


def loglik_order2(obs, P2):
    """Log-likelihood of order-2 Markov chain."""
    T = len(obs)
    loglik = 0.0
    for t in range(2, T):
        context = (obs[t - 2], obs[t - 1])
        nxt = obs[t]
        prob = P2[context][nxt]
        loglik += np.log(prob + 1e-15)
    return loglik


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================
def analyze_spectrum(P, K, verifier=None, name="P"):
    """Eigenvalues, eigenvectors, and derived metrics."""
    eigvals, eigvecs = np.linalg.eig(P)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    abs_eig = np.abs(eigvals)

    eigvals_l, eigvecs_l = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals_l - 1))
    pi = np.real(eigvecs_l[:, idx])
    pi = np.abs(pi)
    pi = pi / pi.sum()

    H = -np.sum(pi[:, None] * P * np.log(P + 1e-15))
    H_max = np.log(K)

    if abs_eig[1] >= 1:
        t_mix = np.inf
    elif abs_eig[1] <= 0:
        t_mix = 1.0
    else:
        t_mix = np.log(0.01) / np.log(abs_eig[1])

    contrib = abs_eig / abs_eig.sum()
    effective_rank = 1.0 / np.sum(contrib ** 2)

    if verifier is not None:
        verifier.check(f"{name}: |lambda1| = 1",
                       np.isclose(abs_eig[0], 1.0, atol=1e-10),
                       f"|lambda1| = {abs_eig[0]:.12f}")
        verifier.check(f"{name}: |lambda_i| <= 1",
                       np.all(abs_eig <= 1.0 + 1e-10),
                       f"max |lambda_i| = {abs_eig.max():.12f}")
        verifier.check(f"{name}: pi sums to 1",
                       np.isclose(pi.sum(), 1.0, atol=1e-10),
                       f"sum = {pi.sum():.12f}")
        verifier.check(f"{name}: pi >= 0", np.all(pi >= 0),
                       f"min pi = {pi.min():.2e}")
        piP = pi @ P
        fixed_point_error = np.max(np.abs(piP - pi))
        verifier.check(f"{name}: piP = pi", fixed_point_error < 1e-8,
                       f"max error = {fixed_point_error:.2e}")
        verifier.check(f"{name}: 0 <= H <= log K",
                       0 <= H <= H_max + 1e-10,
                       f"H = {H:.6f}, log K = {H_max:.6f}")
        ratio = H / H_max if H_max > 0 else 0
        verifier.check(f"{name}: H/H_max in [0,1]",
                       0 <= ratio <= 1 + 1e-10,
                       f"ratio = {ratio:.6f}")
        P_unif = np.ones((K, K)) / K
        pi_unif = np.ones(K) / K
        H_unif = -np.sum(pi_unif[:, None] * P_unif *
                         np.log(P_unif + 1e-15))
        verifier.check(f"{name}: entropy of uniform matrix = log K",
                       np.isclose(H_unif, H_max, atol=1e-10),
                       f"H_unif={H_unif:.8f}, log K={H_max:.8f}")
        verifier.check(f"{name}: irreducible (|lambda2| < 1)",
                       abs_eig[1] < 1.0 - 1e-8,
                       f"|lambda2| = {abs_eig[1]:.10f}")
        try:
            cond_P = np.linalg.cond(P)
            verifier.check(f"{name}: P well conditioned",
                           cond_P < 1e8, f"cond(P) = {cond_P:.3e}")
        except Exception as e:
            verifier.check(f"{name}: P well conditioned", False,
                           f"error: {e}")

    return {
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'abs_eig': abs_eig,
        'pi': pi,
        'entropy': H,
        'entropy_max': H_max,
        'entropy_ratio': H / H_max if H_max > 0 else 0,
        't_mix': t_mix,
        'effective_rank': effective_rank,
        'contrib': contrib,
    }


# ============================================================
# GAP ANALYSIS
# ============================================================
def gaps_by_blocks(gaps, block_size=BLOCK_SIZE,
                   n_bins=N_BINS_GAPS_BLOCKS, verifier=None):
    """Block-wise gap analysis using binned gaps."""
    print("\n" + "=" * 80)
    print("  BLOCK-WISE GAP ANALYSIS")
    print("=" * 80)

    n_blocks = len(gaps) // block_size
    print(f"Number of blocks: {n_blocks} (size = {block_size})")
    print(f"Gaps binned into {n_bins} percentile bins")

    if n_blocks < 5:
        print("  -> Too few blocks for robust analysis.")
        return None

    obs_gaps_bins_global, _ = bin_values(gaps, n_bins)

    lambdas_real = []
    lambdas_cram = []
    rng = np.random.default_rng(SEMILLA)

    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size

        obs_block = obs_gaps_bins_global[start:end]
        P_real, _ = transition_matrix_order1(obs_block, n_bins)
        spec_real = analyze_spectrum(P_real, n_bins)
        lambdas_real.append(spec_real['abs_eig'][1])

        obs_cram = rng.choice(obs_gaps_bins_global, size=len(obs_block),
                              replace=True)
        P_cram, _ = transition_matrix_order1(obs_cram, n_bins)
        spec_cram = analyze_spectrum(P_cram, n_bins)
        lambdas_cram.append(spec_cram['abs_eig'][1])

    lambdas_real = np.array(lambdas_real)
    lambdas_cram = np.array(lambdas_cram)

    mean_real = lambdas_real.mean()
    mean_cram = lambdas_cram.mean()
    diff = mean_real - mean_cram

    print(f"\n  Mean |lambda2| across blocks:")
    print(f"    Real:   {mean_real:.5f} +/- {lambdas_real.std():.5f}")
    print(f"    Cramer: {mean_cram:.5f} +/- {lambdas_cram.std():.5f}")
    print(f"    Delta (real - Cramer): {diff:+.5f}")

    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(lambdas_real, lambdas_cram)
    except ImportError:
        t_stat, p_value = 0.0, 1.0

    print(f"\n  Paired t-test:")
    print(f"    t = {t_stat:.3f}")
    print(f"    p-value = {p_value:.2e}")

    if p_value < 0.001 and diff > 0.05:
        print("  -> GENUINE STRUCTURE: real gaps have more structure")
        print("     than Cramer within each block.")
        result = "genuine"
    elif p_value < 0.05 and diff > 0.02:
        print("  -> MODERATE STRUCTURE: partial evidence.")
        result = "moderate"
    else:
        print("  -> NOT SIGNIFICANT: may be a density artifact.")
        result = "not_significant"

    if verifier is not None:
        verifier.check("Block analysis: blocks computed", n_blocks >= 5,
                       f"n_blocks = {n_blocks}")
        verifier.check("Block analysis: |lambda2| in range",
                       np.all((lambdas_real >= 0) & (lambdas_real <= 1)),
                       f"min={lambdas_real.min():.4f}, "
                       f"max={lambdas_real.max():.4f}")
        verifier.check("Block analysis: bins used",
                       n_bins == N_BINS_GAPS_BLOCKS, f"n_bins = {n_bins}")

    return {
        'lambdas_real': lambdas_real,
        'lambdas_cram': lambdas_cram,
        'mean_real': mean_real,
        'mean_cram': mean_cram,
        'diff': diff,
        't_stat': t_stat,
        'p_value': p_value,
        'result': result,
        'n_bins': n_bins,
    }


def gaps_significance_test(gaps, n_shuffles=N_SHUFFLES_GAPS,
                            verifier=None):
    """Significance test of gaps against shuffled gaps."""
    print("\n" + "=" * 80)
    print("  GAP SIGNIFICANCE TEST AGAINST SHUFFLE")
    print("=" * 80)

    gap_values = sorted(set(gaps))
    gap_map = {g: i for i, g in enumerate(gap_values)}
    K_g = len(gap_values)
    obs_gaps = np.array([gap_map[g] for g in gaps])

    P_real, _ = transition_matrix_order1(obs_gaps, K_g)
    spec_real = analyze_spectrum(P_real, K_g)
    lambda_real = spec_real['abs_eig'][1]

    lambdas_shuffle = []
    rng = np.random.default_rng(SEMILLA)
    for _ in range(n_shuffles):
        obs_shuf = obs_gaps.copy()
        rng.shuffle(obs_shuf)
        P_shuf, _ = transition_matrix_order1(obs_shuf, K_g)
        spec_shuf = analyze_spectrum(P_shuf, K_g)
        lambdas_shuffle.append(spec_shuf['abs_eig'][1])

    lambdas_shuffle = np.array(lambdas_shuffle)
    mean = lambdas_shuffle.mean()
    std = lambdas_shuffle.std()
    z = (lambda_real - mean) / std if std > 0 else 0
    p_value = (lambdas_shuffle >= lambda_real).mean()

    print(f"\n  |lambda2| real    = {lambda_real:.5f}")
    print(f"  |lambda2| shuffle = {mean:.5f} +/- {std:.5f}")
    print(f"  Delta (real - shuffle) = {lambda_real - mean:+.5f}")
    print(f"  z-score = {z:.3f}")
    print(f"  p-value = {p_value:.4f}")

    if z > 3:
        print("  -> SIGNIFICANT: real well above shuffle.")
        result = "significant"
    elif z > 2:
        print("  -> MODERATELY SIGNIFICANT.")
        result = "moderate"
    else:
        print("  -> NOT SIGNIFICANT.")
        result = "not_significant"

    if verifier is not None:
        verifier.check("Gap significance: shuffles generated",
                       len(lambdas_shuffle) == n_shuffles,
                       f"n = {len(lambdas_shuffle)}")

    return {
        'lambda_real': lambda_real,
        'lambda_shuffle_mean': mean,
        'lambda_shuffle_std': std,
        'z': z,
        'p_value': p_value,
        'result': result,
    }


def normalized_gaps_test(primes, gaps, verifier=None):
    """Normalized gaps vs refined Cramer normalized by local density."""
    print("\n" + "=" * 80)
    print("  NORMALIZED GAP ANALYSIS VS REFINED CRAMER")
    print("=" * 80)

    primes_arr = np.array(primes)
    real_gaps = np.diff(primes_arr)
    log_primes = np.log(primes_arr[:-1])
    gaps_norm = real_gaps / log_primes

    print(f"\n  Statistics of real normalized gaps:")
    print(f"    Mean: {gaps_norm.mean():.5f}")
    print(f"    Std:  {gaps_norm.std():.5f}")
    print(f"    Min:  {gaps_norm.min():.5f}")
    print(f"    Max:  {gaps_norm.max():.5f}")

    print("\n  Generating pseudo-primes with observed gap distribution...")
    primes_cr = generate_pseudo_primes_with_gaps(len(primes), real_gaps,
                                                  SEMILLA + 1)
    gaps_cr = np.diff(primes_cr)
    log_cr = np.log(primes_cr[:-1])
    gaps_cr_norm = gaps_cr / log_cr

    print(f"\n  Statistics of refined Cramer normalized gaps:")
    print(f"    Mean: {gaps_cr_norm.mean():.5f}")
    print(f"    Std:  {gaps_cr_norm.std():.5f}")
    print(f"    Min:  {gaps_cr_norm.min():.5f}")
    print(f"    Max:  {gaps_cr_norm.max():.5f}")

    n_bins = N_BINS_GAPS_KS
    all_vals = np.concatenate([gaps_norm, gaps_cr_norm])
    bins = np.percentile(all_vals, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-10
    bins[-1] += 1e-10

    obs_real_bins, _ = bin_values(gaps_norm, n_bins, bins=bins)
    obs_cr_bins, _ = bin_values(gaps_cr_norm, n_bins, bins=bins)

    P_real, _ = transition_matrix_order1(obs_real_bins, n_bins)
    spec_real = analyze_spectrum(P_real, n_bins)

    P_cr, _ = transition_matrix_order1(obs_cr_bins, n_bins)
    spec_cr = analyze_spectrum(P_cr, n_bins)

    diff = spec_real['abs_eig'][1] - spec_cr['abs_eig'][1]

    print(f"\n  Spectral analysis (after normalization):")
    print(f"    |lambda2| real           = {spec_real['abs_eig'][1]:.5f}")
    print(f"    |lambda2| refined Cramer = {spec_cr['abs_eig'][1]:.5f}")
    print(f"    Delta (real - Cramer) = {diff:+.5f}")

    if diff > 0.05:
        print("  -> GENUINE STRUCTURE PERSISTS after normalization.")
        result = "persists"
    elif diff > 0.01:
        print("  -> Weak structure persists. Ambiguous.")
        result = "weak"
    else:
        print("  -> Structure disappears after normalization.")
        result = "disappears"

    if verifier is not None:
        verifier.check("Normalized gaps: real computed",
                       len(gaps_norm) == len(real_gaps),
                       f"len = {len(gaps_norm)}")
        verifier.check("Normalized gaps: refined Cramer computed",
                       len(gaps_cr_norm) == len(gaps_cr),
                       f"len = {len(gaps_cr_norm)}")
        verifier.check("Normalized gaps: bins in range (real)",
                       np.all((obs_real_bins >= 0) &
                              (obs_real_bins < n_bins)),
                       f"min={obs_real_bins.min()}, "
                       f"max={obs_real_bins.max()}")
        verifier.check("Normalized gaps: bins in range (Cramer)",
                       np.all((obs_cr_bins >= 0) &
                              (obs_cr_bins < n_bins)),
                       f"min={obs_cr_bins.min()}, "
                       f"max={obs_cr_bins.max()}")

    return {
        'gaps_norm': gaps_norm,
        'gaps_cr_norm': gaps_cr_norm,
        'spec_real': spec_real,
        'spec_cr': spec_cr,
        'diff': diff,
        'result': result,
    }


def residue_gap_correlation(primes, modulo, verifier=None):
    """Residue-gap correlation normalized by local density."""
    print("\n" + "=" * 80)
    print(f"  RESIDUE-GAP CORRELATION (mod {modulo})")
    print("=" * 80)

    primes_arr = np.array(primes)
    gaps = np.diff(primes_arr)
    log_primes = np.log(primes_arr[:-1])
    residues = primes_arr[:-1] % modulo

    gaps_norm = gaps / log_primes

    try:
        from scipy import stats
        corr, p_value = stats.pearsonr(residues, gaps_norm)
    except ImportError:
        corr, p_value = np.corrcoef(residues, gaps_norm)[0, 1], 0.0

    print(f"\n  Pearson correlation (residue, gap/log p):")
    print(f"    r = {corr:.5f}")
    print(f"    p-value = {p_value:.2e}")

    coprimes = [r for r in range(modulo) if np.gcd(r, modulo) == 1]
    gaps_by_class = defaultdict(list)
    for i, p in enumerate(primes_arr[:-1]):
        if np.gcd(p, modulo) == 1:
            gaps_by_class[p % modulo].append(gaps_norm[i])

    mean_by_class = {}
    for r in coprimes:
        if gaps_by_class[r]:
            mean_by_class[r] = np.mean(gaps_by_class[r])

    if mean_by_class:
        values = list(mean_by_class.values())
        print(f"\n  Mean normalized gap by residue class mod {modulo}:")
        print(f"    Global mean: {np.mean(values):.5f}")
        print(f"    Std across classes: {np.std(values):.5f}")
        print(f"    Min: {np.min(values):.5f} (residue "
              f"{min(mean_by_class, key=mean_by_class.get)})")
        print(f"    Max: {np.max(values):.5f} (residue "
              f"{max(mean_by_class, key=mean_by_class.get)})")

        groups = [gaps_by_class[r] for r in coprimes if gaps_by_class[r]]
        try:
            from scipy import stats
            f_stat, p_anova = stats.f_oneway(*groups)
        except ImportError:
            f_stat, p_anova = 0.0, 1.0

        print(f"\n  ANOVA (means across classes, normalized gaps):")
        print(f"    F = {f_stat:.3f}")
        print(f"    p-value = {p_anova:.2e}")

        if p_anova < 0.001:
            print("  -> SIGNIFICANT DIFFERENCES across classes.")
            result = "significant"
        elif p_anova < 0.05:
            print("  -> Moderate differences.")
            result = "moderate"
        else:
            print("  -> No significant differences.")
            result = "not_significant"
    else:
        f_stat, p_anova, result = 0, 1, "no_data"

    if verifier is not None:
        verifier.check(f"Residue-gap correlation mod {modulo}",
                       -1 <= corr <= 1, f"r = {corr:.5f}")
        verifier.check(f"ANOVA mod {modulo}",
                       0 <= p_anova <= 1, f"p = {p_anova:.2e}")

    return {
        'corr': corr,
        'p_value_corr': p_value,
        'mean_by_class': mean_by_class,
        'f_stat': f_stat,
        'p_anova': p_anova,
        'result': result,
    }


# ============================================================
# EXTENDED EXPLORATION
# ============================================================
def analyze_single_range(primes_range, range_idx, verifier=None):
    """Analyze a single narrow range with normalization."""
    if len(primes_range) < MIN_PRIMES_RANGE:
        return None

    primes_arr = np.array(primes_range)
    gaps_range = np.diff(primes_arr)
    log_range = np.log(primes_arr[:-1])
    gaps_norm = gaps_range / log_range

    n_bins = 20
    bins = np.percentile(gaps_norm, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-10
    bins[-1] += 1e-10

    obs_real, _ = bin_values(gaps_norm, n_bins, bins=bins)
    P_real, _ = transition_matrix_order1(obs_real, n_bins)
    spec_real = analyze_spectrum(P_real, n_bins)

    rng = np.random.default_rng(SEMILLA + range_idx)
    gap_values, counts = np.unique(gaps_range, return_counts=True)
    probs = counts / counts.sum()

    p_current = int(primes_arr[0])
    cr_primes = [p_current]
    while len(cr_primes) < len(primes_arr):
        gap = rng.choice(gap_values, p=probs)
        p_current += int(gap)
        cr_primes.append(p_current)
    cr_arr = np.array(cr_primes)
    cr_gaps = np.diff(cr_arr)
    cr_log = np.log(cr_arr[:-1])
    cr_gaps_norm = cr_gaps / cr_log

    obs_cr, _ = bin_values(cr_gaps_norm, n_bins, bins=bins)
    P_cr, _ = transition_matrix_order1(obs_cr, n_bins)
    spec_cr = analyze_spectrum(P_cr, n_bins)

    diff = spec_real['abs_eig'][1] - spec_cr['abs_eig'][1]

    return {
        'range_idx': range_idx,
        'n_primes': len(primes_range),
        'lambda_real': spec_real['abs_eig'][1],
        'lambda_cr': spec_cr['abs_eig'][1],
        'diff': diff,
    }


def gap_analysis_narrow_ranges(primes, narrow_ranges=NARROW_RANGES,
                                verifier=None):
    """Gap structure within 10 narrow ranges."""
    print("\n" + "=" * 80)
    print("  EXTENDED: GAP ANALYSIS ACROSS 10 NARROW RANGES")
    print("=" * 80)

    primes_arr = np.array(primes)
    results = []

    for idx, (lo, hi) in enumerate(narrow_ranges):
        mask = (primes_arr >= lo) & (primes_arr < hi)
        primes_range = primes_arr[mask]

        print(f"\n  Range [{lo:,}, {hi:,}]: "
              f"{len(primes_range):,} primes")

        if len(primes_range) < MIN_PRIMES_RANGE:
            print(f"    Too few primes (< {MIN_PRIMES_RANGE}), skipping.")
            continue

        res = analyze_single_range(primes_range, idx, verifier=verifier)
        if res is None:
            print("    Analysis returned None, skipping.")
            continue

        print(f"    |lambda2| real           = {res['lambda_real']:.5f}")
        print(f"    |lambda2| refined Cramer = {res['lambda_cr']:.5f}")
        print(f"    Delta = {res['diff']:+.5f}")

        results.append({
            'range': (lo, hi),
            'n_primes': res['n_primes'],
            'lambda_real': res['lambda_real'],
            'lambda_cr': res['lambda_cr'],
            'diff': res['diff'],
        })

    if verifier is not None:
        verifier.check("Narrow ranges analyzed",
                       len(results) >= 5,
                       f"n_ranges = {len(results)}")
        if results:
            diffs = [r['diff'] for r in results]
            verifier.check("Narrow ranges: deltas computed",
                           all(np.isfinite(d) for d in diffs),
                           f"min={min(diffs):.5f}, max={max(diffs):.5f}")

    return results


def sliding_window_density_analysis(primes, n_windows=N_WINDOWS_DENSITY,
                                     verifier=None):
    """Sliding-window density normalization with trend test."""
    print("\n" + "=" * 80)
    print("  EXTENDED: SLIDING-WINDOW DENSITY NORMALIZATION")
    print("=" * 80)

    primes_arr = np.array(primes)
    gaps = np.diff(primes_arr)
    log_primes = np.log(primes_arr[:-1])
    gaps_norm = gaps / log_primes

    n = len(gaps_norm)
    window_size = n // n_windows

    print(f"  Number of windows: {n_windows}")
    print(f"  Window size: {window_size}")

    n_bins = 20
    bins = np.percentile(gaps_norm, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-10
    bins[-1] += 1e-10

    lambdas = []
    means = []
    stds = []

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window = gaps_norm[start:end]

        if len(window) < 100:
            continue

        obs_window, _ = bin_values(window, n_bins, bins=bins)
        P_window, _ = transition_matrix_order1(obs_window, n_bins)
        spec_window = analyze_spectrum(P_window, n_bins)

        lambdas.append(spec_window['abs_eig'][1])
        means.append(window.mean())
        stds.append(window.std())

    lambdas = np.array(lambdas)
    means = np.array(means)
    stds = np.array(stds)

    print(f"\n  Across windows:")
    print(f"    Mean of window means:  {means.mean():.5f}")
    print(f"    Std of window means:   {means.std():.5f}")
    print(f"    Mean of window stds:   {stds.mean():.5f}")
    print(f"    Std of window stds:    {stds.std():.5f}")
    print(f"    Mean |lambda2|:        {lambdas.mean():.5f}")
    print(f"    Std |lambda2|:         {lambdas.std():.5f}")
    print(f"    Min |lambda2|:         {lambdas.min():.5f}")
    print(f"    Max |lambda2|:         {lambdas.max():.5f}")

    slope, intercept, r_value, p_value, std_err = 0.0, 0.0, 0.0, 1.0, 0.0
    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(np.arange(len(lambdas)), lambdas)
        print(f"\n  Linear trend of |lambda2| across windows:")
        print(f"    Slope = {slope:.6f}")
        print(f"    R^2 = {r_value ** 2:.5f}")
        print(f"    p-value = {p_value:.2e}")
        if p_value < 0.05:
            print("  -> Significant trend: structure changes across sequence.")
            result = "non_stationary"
        else:
            print("  -> No significant trend: structure is stationary.")
            result = "stationary"
    except ImportError:
        result = "unknown"

    if verifier is not None:
        verifier.check("Sliding windows analyzed",
                       len(lambdas) == n_windows,
                       f"n_windows = {len(lambdas)}")
        verifier.check("Sliding windows: |lambda2| in range",
                       np.all((lambdas >= 0) & (lambdas <= 1)),
                       f"min={lambdas.min():.4f}, "
                       f"max={lambdas.max():.4f}")
        verifier.check("Sliding windows: trend computed",
                       np.isfinite(slope) and 0 <= p_value <= 1,
                       f"slope={slope:.6f}, p={p_value:.2e}")

    return {
        'lambdas': lambdas,
        'means': means,
        'stds': stds,
        'slope': slope,
        'p_value_trend': p_value,
        'result': result,
    }


# ============================================================
# ADDITIONAL TESTS
# ============================================================
def stationarity_test_gaps(gaps, verifier=None):
    """Stationarity test: first half vs second half of gaps."""
    print("\n" + "=" * 80)
    print("  STATIONARITY TEST OF GAPS")
    print("=" * 80)

    n = len(gaps)
    half = n // 2
    gaps_1 = gaps[:half]
    gaps_2 = gaps[half:]

    n_bins = 20
    bins = np.percentile(gaps, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-10
    bins[-1] += 1e-10

    obs_1 = np.clip(np.digitize(gaps_1, bins) - 1, 0, n_bins - 1)
    obs_2 = np.clip(np.digitize(gaps_2, bins) - 1, 0, n_bins - 1)

    P_1, _ = transition_matrix_order1(obs_1, n_bins)
    P_2, _ = transition_matrix_order1(obs_2, n_bins)

    spec_1 = analyze_spectrum(P_1, n_bins)
    spec_2 = analyze_spectrum(P_2, n_bins)

    l1 = spec_1['abs_eig'][1]
    l2 = spec_2['abs_eig'][1]
    diff = abs(l1 - l2)

    print(f"\n  |lambda2| first half:  {l1:.5f}")
    print(f"  |lambda2| second half: {l2:.5f}")
    print(f"  Absolute difference:   {diff:.5f}")

    if diff < 0.02:
        print("  -> STATIONARY: both halves have the same structure.")
        result = "stationary"
    elif diff < 0.05:
        print("  -> SLIGHTLY NON-STATIONARY.")
        result = "slight"
    else:
        print("  -> NON-STATIONARY: structure changes with position.")
        result = "non_stationary"

    if verifier is not None:
        verifier.check("Gap stationarity: |lambda2| computed",
                       np.isfinite(l1) and np.isfinite(l2),
                       f"l1={l1:.5f}, l2={l2:.5f}")

    return {'l1': l1, 'l2': l2, 'diff': diff, 'result': result}


def stationarity_test_residues(obs, K, verifier=None, name="residues"):
    """Stationarity test of residues: first half vs second half."""
    print("\n" + "=" * 80)
    print(f"  STATIONARITY TEST OF {name.upper()}")
    print("=" * 80)

    n = len(obs)
    half = n // 2
    obs_1 = obs[:half]
    obs_2 = obs[half:]

    P_1, _ = transition_matrix_order1(obs_1, K)
    P_2, _ = transition_matrix_order1(obs_2, K)

    spec_1 = analyze_spectrum(P_1, K)
    spec_2 = analyze_spectrum(P_2, K)

    l1 = spec_1['abs_eig'][1]
    l2 = spec_2['abs_eig'][1]
    diff = abs(l1 - l2)

    print(f"\n  |lambda2| first half:  {l1:.5f}")
    print(f"  |lambda2| second half: {l2:.5f}")
    print(f"  Absolute difference:   {diff:.5f}")

    if diff < 0.02:
        print("  -> STATIONARY.")
        result = "stationary"
    elif diff < 0.05:
        print("  -> SLIGHTLY NON-STATIONARY.")
        result = "slight"
    else:
        print("  -> NON-STATIONARY.")
        result = "non_stationary"

    if verifier is not None:
        verifier.check(f"Stationarity {name}: |lambda2| computed",
                       np.isfinite(l1) and np.isfinite(l2),
                       f"l1={l1:.5f}, l2={l2:.5f}")

    return {'l1': l1, 'l2': l2, 'diff': diff, 'result': result}


def ks_test_normalized_gaps(gaps, log_primes, verifier=None):
    """KS test: normalized gaps vs exponential."""
    print("\n" + "=" * 80)
    print("  KS TEST OF NORMALIZED GAPS AGAINST EXPONENTIAL")
    print("=" * 80)

    gaps_norm = gaps / log_primes

    try:
        from scipy import stats
        ks_stat, p_value = stats.kstest(gaps_norm, 'expon')
    except ImportError:
        print("  -> scipy not available, skipping KS test.")
        return None

    print(f"\n  KS statistic: {ks_stat:.5f}")
    print(f"  p-value:      {p_value:.2e}")

    if p_value < 0.001:
        print("  -> REJECTS exponential: structure not explained")
        print("     by density alone.")
        result = "rejects"
    elif p_value < 0.05:
        print("  -> Weakly rejects.")
        result = "weak"
    else:
        print("  -> Does NOT reject: consistent with exponential.")
        result = "consistent"

    if verifier is not None:
        verifier.check("KS: statistic computed",
                       0 <= ks_stat <= 1, f"KS = {ks_stat:.5f}")

    return {'ks_stat': ks_stat, 'p_value': p_value, 'result': result}


# ============================================================
# EXTENSIONS
# ============================================================
def compare_refined_cramer(obs_real, obs_cram_ref, obs_cram_classical, K,
                            verifier=None):
    """Compare real vs classical Cramer vs refined Cramer."""
    print("\n" + "=" * 80)
    print("  REFINED CRAMER COMPARISON (density + observed gaps)")
    print("=" * 80)

    P_real, _ = transition_matrix_order1(obs_real, K, verifier=verifier,
                                          name="P_real")
    P_cc, _ = transition_matrix_order1(obs_cram_classical, K,
                                        name="P_cram_classical")
    P_cr, _ = transition_matrix_order1(obs_cram_ref, K,
                                        name="P_cram_refined")

    spec_real = analyze_spectrum(P_real, K, verifier=verifier,
                                  name="spec_real")
    spec_cc = analyze_spectrum(P_cc, K, name="spec_cc")
    spec_cr = analyze_spectrum(P_cr, K, name="spec_cr")

    print(f"\n{'Metric':>25} | {'real':>10} | {'Cram class':>10} | "
          f"{'Cram ref':>10}")
    print("-" * 65)
    print(f"{'|lambda2|':>25} | {spec_real['abs_eig'][1]:>10.5f} | "
          f"{spec_cc['abs_eig'][1]:>10.5f} | {spec_cr['abs_eig'][1]:>10.5f}")
    print(f"{'Entropy H':>25} | {spec_real['entropy']:>10.4f} | "
          f"{spec_cc['entropy']:>10.4f} | {spec_cr['entropy']:>10.4f}")
    print(f"{'H / H_max':>25} | {spec_real['entropy_ratio']:>10.4f} | "
          f"{spec_cc['entropy_ratio']:>10.4f} | "
          f"{spec_cr['entropy_ratio']:>10.4f}")
    print(f"{'Effective rank':>25} | {spec_real['effective_rank']:>10.2f} | "
          f"{spec_cc['effective_rank']:>10.2f} | "
          f"{spec_cr['effective_rank']:>10.2f}")
    print(f"{'Mixing time':>25} | {spec_real['t_mix']:>10.2f} | "
          f"{spec_cc['t_mix']:>10.2f} | {spec_cr['t_mix']:>10.2f}")

    diff_ref = spec_real['abs_eig'][1] - spec_cr['abs_eig'][1]
    diff_classical = spec_real['abs_eig'][1] - spec_cc['abs_eig'][1]
    print(f"\nDelta |lambda2| (real - classical Cramer)  = "
          f"{diff_classical:+.5f}")
    print(f"Delta |lambda2| (real - refined Cramer) = {diff_ref:+.5f}")

    if abs(diff_ref) < 0.02:
        print("-> Refined Cramer also reproduces real structure.")
    elif diff_ref > 0.05:
        print("-> Real has MORE structure than refined Cramer.")
    else:
        print("-> Ambiguous result.")

    return spec_real, spec_cc, spec_cr


def order2_markov_test(obs_real, obs_shuf, obs_cram, K):
    """Order-2 Markov chain test."""
    print("\n" + "=" * 80)
    print("  ORDER-2 MARKOV CHAIN TEST")
    print("=" * 80)

    P1_real, _ = transition_matrix_order1(obs_real, K)
    ll1_real = sum(np.log(P1_real[obs_real[t-1], obs_real[t]] + 1e-15)
                   for t in range(1, len(obs_real)))

    P1_shuf, _ = transition_matrix_order1(obs_shuf, K)
    ll1_shuf = sum(np.log(P1_shuf[obs_shuf[t-1], obs_shuf[t]] + 1e-15)
                   for t in range(1, len(obs_shuf)))

    P1_cram, _ = transition_matrix_order1(obs_cram, K)
    ll1_cram = sum(np.log(P1_cram[obs_cram[t-1], obs_cram[t]] + 1e-15)
                   for t in range(1, len(obs_cram)))

    P2_real, _ = transition_matrix_order2(obs_real, K)
    ll2_real = loglik_order2(obs_real, P2_real)

    P2_shuf, _ = transition_matrix_order2(obs_shuf, K)
    ll2_shuf = loglik_order2(obs_shuf, P2_shuf)

    P2_cram, _ = transition_matrix_order2(obs_cram, K)
    ll2_cram = loglik_order2(obs_cram, P2_cram)

    n1 = len(obs_real) - 1
    n2 = len(obs_real) - 2

    print(f"\n{'Model':>15} | {'ll total':>15} | {'ll/obs':>12}")
    print("-" * 50)
    print(f"{'order1 real':>15} | {ll1_real:>15.2f} | {ll1_real/n1:>12.5f}")
    print(f"{'order1 shuffle':>15} | {ll1_shuf:>15.2f} | {ll1_shuf/n1:>12.5f}")
    print(f"{'order1 Cramer':>15} | {ll1_cram:>15.2f} | {ll1_cram/n1:>12.5f}")
    print(f"{'order2 real':>15} | {ll2_real:>15.2f} | {ll2_real/n2:>12.5f}")
    print(f"{'order2 shuffle':>15} | {ll2_shuf:>15.2f} | {ll2_shuf/n2:>12.5f}")
    print(f"{'order2 Cramer':>15} | {ll2_cram:>15.2f} | {ll2_cram/n2:>12.5f}")

    delta_rs_1 = (ll1_real - ll1_shuf) / n1
    delta_rs_2 = (ll2_real - ll2_shuf) / n2
    delta_rc_1 = (ll1_real - ll1_cram) / n1
    delta_rc_2 = (ll2_real - ll2_cram) / n2

    print(f"\nDelta ll/obs (real - shuffle), order 1: {delta_rs_1:+.5f}")
    print(f"Delta ll/obs (real - shuffle), order 2: {delta_rs_2:+.5f}")
    print(f"Delta ll/obs (real - Cramer), order 1:  {delta_rc_1:+.5f}")
    print(f"Delta ll/obs (real - Cramer), order 2:  {delta_rc_2:+.5f}")

    if delta_rs_2 > delta_rs_1:
        print("-> Order 2 captures MORE structure than order 1.")
    else:
        print("-> Order 2 does NOT capture more structure: order-1 memory.")

    return {
        'll1_real': ll1_real, 'll1_shuf': ll1_shuf, 'll1_cram': ll1_cram,
        'll2_real': ll2_real, 'll2_shuf': ll2_shuf, 'll2_cram': ll2_cram,
        'delta_rs_1': delta_rs_1, 'delta_rs_2': delta_rs_2,
        'delta_rc_1': delta_rc_1, 'delta_rc_2': delta_rc_2,
    }


def analyze_large_modulus(primes, modulo, verifier=None):
    """Spectral analysis for a large modulus."""
    obs_real, coprimes = residues_to_indices(primes, modulo)
    K = len(coprimes)
    print(f"\nModulus {modulo}: K = {K} symbols, "
          f"{len(obs_real)} observations")

    if len(obs_real) < 1000:
        print("  -> Too few data points for such a large modulus.")
        return None

    obs_shuf = shuffle_observations(obs_real, SEMILLA)
    obs_cram = generate_cramer_classical(len(obs_real), modulo, SEMILLA)

    P_real, _ = transition_matrix_order1(obs_real, K)
    P_shuf, _ = transition_matrix_order1(obs_shuf, K)
    P_cram, _ = transition_matrix_order1(obs_cram, K)

    spec_real = analyze_spectrum(P_real, K)
    spec_shuf = analyze_spectrum(P_shuf, K)
    spec_cram = analyze_spectrum(P_cram, K)

    print(f"  |lambda2| real    = {spec_real['abs_eig'][1]:.5f}")
    print(f"  |lambda2| shuffle = {spec_shuf['abs_eig'][1]:.5f}")
    print(f"  |lambda2| Cramer  = {spec_cram['abs_eig'][1]:.5f}")
    print(f"  H/H_max real = {spec_real['entropy_ratio']:.5f}")
    print(f"  Effective rank real = {spec_real['effective_rank']:.2f}")

    return {
        'modulo': modulo,
        'K': K,
        'spec_real': spec_real,
        'spec_shuf': spec_shuf,
        'spec_cram': spec_cram,
    }


# ============================================================
# FIGURE GENERATION
# ============================================================
def figure1_spectrum(res_by_modulus):
    """Figure 1: spectrum real vs shuffle vs classical/refined Cramer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, modulo in zip(axes, [210, 2310]):
        data = res_by_modulus[modulo]
        spec_real = data['spec_real']
        spec_shuf = data['spec_shuf']
        spec_cc = data['spec_cc']
        spec_cr = data['spec_cr']

        n_show = 15
        ax.plot(spec_real['abs_eig'][:n_show], 'o-', label='real',
                markersize=5, color='C0', linewidth=1.5)
        ax.plot(spec_shuf['abs_eig'][:n_show], 'd--', label='shuffle',
                markersize=5, alpha=0.7, color='C1', linewidth=1.2)
        ax.plot(spec_cc['abs_eig'][:n_show], 's--',
                label='classical Cramer',
                markersize=5, alpha=0.8, color='C2', linewidth=1.2)
        ax.plot(spec_cr['abs_eig'][:n_show], '^--',
                label='refined Cramer',
                markersize=5, alpha=0.8, color='C3', linewidth=1.2)

        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel(r'$|\lambda_i|$')
        ax.set_title(f'Modulus {modulo}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig('figure1_spectrum.png', dpi=300)
    plt.savefig('figure1_spectrum.pdf')
    plt.close()
    print("Saved: figure1_spectrum.png / .pdf")


def figure2_gaps(gap_info, primes):
    """Figure 2: gap spectrum and distribution of normalized gaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    spec_real = gap_info['spec_gaps']
    spec_shuf = gap_info['spec_gaps_shuf']
    spec_cram = gap_info['spec_gaps_cram']

    n_show = 20
    axes[0].plot(spec_real['abs_eig'][:n_show], 'o-',
                 label='real gaps',
                 markersize=5, color='C0', linewidth=1.5)
    axes[0].plot(spec_shuf['abs_eig'][:n_show], 'd--',
                 label='shuffled gaps',
                 markersize=5, alpha=0.7, color='C1', linewidth=1.2)
    axes[0].plot(spec_cram['abs_eig'][:n_show], 's--',
                 label='Cramer gaps',
                 markersize=5, alpha=0.7, color='C2', linewidth=1.2)
    axes[0].set_xlabel('Eigenvalue index')
    axes[0].set_ylabel(r'$|\lambda_i|$')
    axes[0].set_title('Gap spectrum')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)

    primes_arr = np.array(primes)
    gaps_real = np.diff(primes_arr)
    log_primes = np.log(primes_arr[:-1])
    gaps_norm = gaps_real / log_primes

    axes[1].hist(gaps_norm, bins=60, density=True, alpha=0.6,
                 color='C0', label='real normalized gaps')

    x_exp = np.linspace(0, np.percentile(gaps_norm, 99.5), 200)
    axes[1].plot(x_exp, np.exp(-x_exp), 'r-', linewidth=2,
                 label=r'exponential ($e^{-x}$)')

    axes[1].set_xlabel(r'$g_n / \log p_n$')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of normalized gaps')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure2_gaps.png', dpi=300)
    plt.savefig('figure2_gaps.pdf')
    plt.close()
    print("Saved: figure2_gaps.png / .pdf")


def figure3_blocks(gap_info):
    """Figure 3: block-wise analysis of gap |lambda2|."""
    if gap_info.get('deep_analysis', {}).get('blocks') is None:
        print("No block data available for figure 3.")
        return

    blocks = gap_info['deep_analysis']['blocks']
    lambdas_real = blocks['lambdas_real']
    lambdas_cram = blocks['lambdas_cram']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 0.4, 30)
    axes[0].hist(lambdas_real, bins=bins, alpha=0.6,
                 color='C0', label='real blocks',
                 density=True)
    axes[0].hist(lambdas_cram, bins=bins, alpha=0.6,
                 color='C2', label='Cramer blocks',
                 density=True)
    axes[0].axvline(lambdas_real.mean(), color='C0',
                    linestyle='--', linewidth=2)
    axes[0].axvline(lambdas_cram.mean(), color='C2',
                    linestyle='--', linewidth=2)
    axes[0].set_xlabel(r'$|\lambda_2|$ within block')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Block-wise gap structure')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[0].text(
        0.98, 0.97,
        f"Real mean = {lambdas_real.mean():.4f}\n"
        f"Cramer mean = {lambdas_cram.mean():.4f}\n"
        f"Delta = {blocks['diff']:+.4f}\n"
        f"t = {blocks['t_stat']:.2f}\n"
        f"p = {blocks['p_value']:.2e}",
        transform=axes[0].transAxes,
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    axes[1].scatter(lambdas_cram, lambdas_real, s=20, alpha=0.6,
                    color='C0')
    axes[1].plot([0, 0.4], [0, 0.4], 'k--', alpha=0.4,
                 label='y = x')
    axes[1].set_xlabel(r'Cramer $|\lambda_2|$')
    axes[1].set_ylabel(r'Real $|\lambda_2|$')
    axes[1].set_title('Per-block comparison')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure3_blocks.png', dpi=300)
    plt.savefig('figure3_blocks.pdf')
    plt.close()
    print("Saved: figure3_blocks.png / .pdf")


def figure4_normalized(gap_info, narrow_results):
    """Figure 4: normalized gap analysis (global and narrow ranges)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    normalized = gap_info.get('deep_analysis', {}).get('normalized', {})
    if normalized:
        spec_real = normalized['spec_real']
        spec_cr = normalized['spec_cr']

        n_show = 15
        axes[0].plot(spec_real['abs_eig'][:n_show], 'o-',
                     label='real normalized',
                     markersize=5, color='C0', linewidth=1.5)
        axes[0].plot(spec_cr['abs_eig'][:n_show], '^--',
                     label='refined Cramer normalized',
                     markersize=5, alpha=0.8, color='C3', linewidth=1.2)
        axes[0].set_xlabel('Eigenvalue index')
        axes[0].set_ylabel(r'$|\lambda_i|$')
        axes[0].set_title(
            f'Normalized gaps (global): '
            f'Delta = {normalized["diff"]:+.4f}')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(alpha=0.3)

    if narrow_results:
        labels = []
        deltas = []
        for r in narrow_results:
            lo, hi = r['range']
            label = f"$[{lo:.0e}, {hi:.0e}]$"
            labels.append(label)
            deltas.append(r['diff'])

        colors = ['C3' if d < 0 else 'C2' for d in deltas]
        x_pos = np.arange(len(labels))
        axes[1].bar(x_pos, deltas, color=colors, alpha=0.7)
        axes[1].axhline(0, color='k', linewidth=1)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=30, ha='right',
                                 fontsize=8)
        axes[1].set_ylabel(r'$\Delta |\lambda_2|$ (real $-$ Cramer)')
        axes[1].set_title('Normalized gap analysis across ranges')
        axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figure4_normalized.png', dpi=300)
    plt.savefig('figure4_normalized.pdf')
    plt.close()
    print("Saved: figure4_normalized.png / .pdf")


def figure5_sliding(sliding_results):
    """Figure 5: sliding-window analysis."""
    if sliding_results is None:
        print("No sliding-window data for figure 5.")
        return

    lambdas = sliding_results['lambdas']
    means = sliding_results['means']
    stds = sliding_results['stds']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(lambdas))
    axes[0].plot(x, lambdas, 'o-', markersize=4, color='C0',
                 linewidth=1.2, label=r'$|\lambda_2|$ per window')
    axes[0].axhline(lambdas.mean(), color='k', linestyle='--',
                    linewidth=1.5,
                    label=f'mean = {lambdas.mean():.4f}')

    slope = sliding_results['slope']
    intercept = lambdas.mean() - slope * x.mean()
    axes[0].plot(x, slope * x + intercept, 'r-', linewidth=2,
                 label=f'slope = {slope:.2e}\np = '
                       f'{sliding_results["p_value_trend"]:.2f}')

    axes[0].set_xlabel('Window index')
    axes[0].set_ylabel(r'$|\lambda_2|$')
    axes[0].set_title('Sliding-window density normalization')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, means, 'o-', markersize=4, color='C0',
                 linewidth=1.2, label='window mean')
    axes[1].errorbar(x, means, yerr=stds, fmt='none',
                     ecolor='C0', alpha=0.3, capsize=2)
    axes[1].axhline(means.mean(), color='k', linestyle='--',
                    linewidth=1.5,
                    label=f'global mean = {means.mean():.4f}')
    axes[1].set_xlabel('Window index')
    axes[1].set_ylabel(r'Mean normalized gap')
    axes[1].set_title('Window means and stds')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure5_sliding.png', dpi=300)
    plt.savefig('figure5_sliding.pdf')
    plt.close()
    print("Saved: figure5_sliding.png / .pdf")


def figure6_summary(res_by_modulus, gap_info, narrow_results):
    """Figure 6: summary of key spectral results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = []
    values_real = []
    values_null = []

    categories.append('Residues\nmod 210')
    values_real.append(res_by_modulus[210]['comparison']['lambda2_real'])
    values_null.append(res_by_modulus[210]['comparison']['lambda2_cr'])

    categories.append('Residues\nmod 2310')
    values_real.append(res_by_modulus[2310]['comparison']['lambda2_real'])
    values_null.append(res_by_modulus[2310]['comparison']['lambda2_cr'])

    categories.append('Gaps\n(global)')
    values_real.append(gap_info['spec_gaps']['abs_eig'][1])
    values_null.append(gap_info['spec_gaps_cram']['abs_eig'][1])

    norm = gap_info.get('deep_analysis', {}).get('normalized', {})
    if norm:
        categories.append('Gaps\n(normalized)')
        values_real.append(norm['spec_real']['abs_eig'][1])
        values_null.append(norm['spec_cr']['abs_eig'][1])

    x = np.arange(len(categories))
    width = 0.35

    bars_real = ax.bar(x - width/2, values_real, width,
                       label='real', color='C0', alpha=0.8)
    bars_null = ax.bar(x + width/2, values_null, width,
                       label='null (Cramer refined)', color='C3',
                       alpha=0.8)

    for bar, val in zip(bars_real, values_real):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars_null, values_null):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel(r'$|\lambda_2|$')
    ax.set_title('Summary of key spectral results')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('figure6_summary.png', dpi=300)
    plt.savefig('figure6_summary.pdf')
    plt.close()
    print("Saved: figure6_summary.png / .pdf")


# ============================================================
# CONCLUSION
# ============================================================
def conclude(results_by_modulus, gap_info, large_moduli,
             narrow_range_results, sliding_window_results):
    print("\n" + "=" * 80)
    print("  AUTOMATIC CONCLUSION")
    print("=" * 80)

    print("\n[1] Order-1 structure in residues:")
    for modulo, data in results_by_modulus.items():
        comp = data['comparison']
        print(f"  mod {modulo}:")
        print(f"    |lambda2| real    = {comp['lambda2_real']:.5f}")
        print(f"    |lambda2| shuffle = {comp['lambda2_shuf']:.5f}")
        print(f"    |lambda2| classical Cramer = {comp['lambda2_cc']:.5f}")
        print(f"    |lambda2| refined Cramer   = {comp['lambda2_cr']:.5f}")
        print(f"    H/H_max real = {comp['H_ratio_real']:.5f}")

    print("\n[2] Refined Cramer (density + gaps):")
    for modulo, data in results_by_modulus.items():
        comp = data['comparison']
        diff_ref = comp['lambda2_real'] - comp['lambda2_cr']
        if abs(diff_ref) < 0.02:
            print(f"  mod {modulo}: refined Cramer REPRODUCES structure.")
        elif diff_ref > 0.05:
            print(f"  mod {modulo}: REAL has MORE structure than refined Cramer.")
        else:
            print(f"  mod {modulo}: ambiguous (Delta = {diff_ref:+.5f}).")

    print("\n[3] Order-2 Markov (per modulus):")
    for modulo, data in results_by_modulus.items():
        if 'order2_info' in data:
            o2 = data['order2_info']
            d1 = o2['delta_rs_1']
            d2 = o2['delta_rs_2']
            print(f"  mod {modulo}: Delta ll/obs order1 = {d1:.5f} | "
                  f"order2 = {d2:.5f}")
            if d2 > d1:
                print(f"    -> Order-2 memory detected.")
            else:
                print(f"    -> Order-1 memory dominates.")

    print("\n[4] Gaps:")
    print(f"  |lambda2| gaps real    = {gap_info['spec_gaps']['abs_eig'][1]:.5f}")
    print(f"  |lambda2| gaps shuffle = {gap_info['spec_gaps_shuf']['abs_eig'][1]:.5f}")
    print(f"  |lambda2| gaps Cramer  = {gap_info['spec_gaps_cram']['abs_eig'][1]:.5f}")
    print(f"  Delta vs shuffle = {gap_info['diff_gaps_shuf']:+.5f}")
    print(f"  Delta vs Cramer  = {gap_info['diff_gaps_cram']:+.5f}")
    if gap_info['diff_gaps_shuf'] > 0.1:
        print("  -> Gaps have STRONG order structure.")
    elif gap_info['diff_gaps_shuf'] > 0.02:
        print("  -> Gaps have MODERATE order structure.")
    else:
        print("  -> Gaps have no significant order structure.")
    if abs(gap_info['diff_gaps_cram']) < 0.02:
        print("  -> Cramer of gaps reproduces structure. Density + gaps enough.")
    elif gap_info['diff_gaps_cram'] > 0.05:
        print("  -> Real gaps have MORE structure than Cramer. Genuine arithmetic.")

    print("\n[5] Large moduli:")
    for res in large_moduli:
        if res:
            print(f"  mod {res['modulo']}: K = {res['K']}, "
                  f"|lambda2| real = {res['spec_real']['abs_eig'][1]:.5f}, "
                  f"|lambda2| Cramer = {res['spec_cram']['abs_eig'][1]:.5f}")

    print("\n[6] Deep gap exploration:")
    op = gap_info.get('deep_analysis', {})
    if op.get('blocks'):
        b = op['blocks']
        print(f"  Block analysis: Delta = {b['diff']:+.5f}, "
              f"p = {b['p_value']:.2e} -> {b['result']}")
    if op.get('significance'):
        s = op['significance']
        print(f"  Significance: z = {s['z']:.3f}, "
              f"p = {s['p_value']:.4f} -> {s['result']}")
    if op.get('normalized'):
        n = op['normalized']
        print(f"  Normalized vs refined Cramer: Delta = "
              f"{n['diff']:+.5f} -> {n['result']}")
    if op.get('corr_210'):
        c = op['corr_210']
        print(f"  Residue-gap mod 210: "
              f"p_ANOVA = {c['p_anova']:.2e} -> {c['result']}")
    if op.get('corr_2310'):
        c = op['corr_2310']
        print(f"  Residue-gap mod 2310: "
              f"p_ANOVA = {c['p_anova']:.2e} -> {c['result']}")

    print("\n[7] Additional tests:")
    if gap_info.get('stationarity_gaps'):
        st = gap_info['stationarity_gaps']
        print(f"  Gap stationarity: l1={st['l1']:.5f}, "
              f"l2={st['l2']:.5f}, diff={st['diff']:.5f} -> "
              f"{st['result']}")
    if gap_info.get('ks'):
        ks = gap_info['ks']
        print(f"  KS exponential: KS={ks['ks_stat']:.5f}, "
              f"p={ks['p_value']:.2e} -> {ks['result']}")

    print("\n[8] Extended exploration (Option B):")
    print("  Narrow ranges (normalized analysis within each range):")
    for r in narrow_range_results:
        lo, hi = r['range']
        print(f"    [{lo:,}, {hi:,}]: n={r['n_primes']:,}, "
              f"|lambda2| real={r['lambda_real']:.5f}, "
              f"Cramer={r['lambda_cr']:.5f}, "
              f"Delta={r['diff']:+.5f}")
    if sliding_window_results:
        sw = sliding_window_results
        print(f"  Sliding-window density normalization:")
        print(f"    Mean |lambda2| = {sw['lambdas'].mean():.5f} "
              f"+/- {sw['lambdas'].std():.5f}")
        print(f"    Trend slope = {sw['slope']:.6f}, "
              f"p = {sw['p_value_trend']:.2e} -> {sw['result']}")

    # Final verdict
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    refined_compatible = all(
        abs(d['comparison']['lambda2_real']
            - d['comparison']['lambda2_cr']) < 0.02
        for d in results_by_modulus.values()
    )
    order2_mayor = any(
        d.get('order2_info', {}).get('delta_rs_2', 0)
        > d.get('order2_info', {}).get('delta_rs_1', 0)
        for d in results_by_modulus.values()
    )
    gaps_vs_cram_strong = gap_info['diff_gaps_cram'] > 0.05

    blocks = op.get('blocks', {}) or {}
    normalized = op.get('normalized', {}) or {}
    stat = gap_info.get('stationarity_gaps', {}) or {}
    ks = gap_info.get('ks', {}) or {}

    blocks_genuine = blocks.get('result') == 'genuine'
    normalized_persists = normalized.get('result') == 'persists'
    stationary = stat.get('result') == 'stationary'
    ks_rejects = ks.get('result') == 'rejects'

    narrow_deltas = [r['diff'] for r in narrow_range_results]
    narrow_all_positive = all(d > 0.01 for d in narrow_deltas) if narrow_deltas else False
    narrow_all_negative = all(d < -0.01 for d in narrow_deltas) if narrow_deltas else False
    sliding_stationary = (sliding_window_results and
                          sliding_window_results.get('result') == 'stationary')

    if refined_compatible and not order2_mayor and not gaps_vs_cram_strong:
        print("""
Prime residues have strong order-1 structure, but that structure is
COMPLETELY reproduced by the refined Cramer model (density 1/log x
+ observed gap distribution). No evidence of:
  - Order-2 memory
  - Gap structure beyond density
  - Modular arithmetic structure at large moduli

The conclusion is that the observable regularity of prime residues
is densitive, not arithmetically deep.
        """)
    elif (gaps_vs_cram_strong and blocks_genuine and normalized_persists
          and ks_rejects):
        print("""
ROBUST FINDING! Gaps have structure that is NOT reproduced by
Cramer, and that structure PERSISTS after:
  - Block analysis with global gap distribution + bins
  - Normalization by local density vs refined Cramer
  - Shuffle comparison
  - KS test against exponential (rejects)
  - Stationarity test

This suggests genuine arithmetic structure in the gaps, not a
density artifact. Potentially publishable.
        """)
    elif gaps_vs_cram_strong:
        print("""
SIGNIFICANT FINDING! Gap structure is NOT reproduced by refined
Cramer. However, the deep exploration does not fully confirm it as
genuine (may be a residual density artifact). More data needed.
        """)
    else:
        print("""
Mixed result. Some evidence of structure not explained by refined
Cramer, but not conclusive. More data or finer moduli needed.
        """)

    print("\n  Extended exploration verdict:")
    if narrow_all_positive:
        print("    Narrow ranges: consistent positive Delta across ranges.")
    elif narrow_all_negative:
        print("    Narrow ranges: consistent negative Delta (structure")
        print("    disappears at all scales).")
    else:
        print("    Narrow ranges: mixed results across scales.")
    if sliding_window_results:
        if sliding_stationary:
            print("    Sliding windows: structure is stationary across sequence.")
        else:
            print("    Sliding windows: structure varies across sequence.")


# ============================================================
# PLOTTING (comparative)
# ============================================================
def plot_comparison(spec_real, spec_cc, spec_cr, spec_shuf, modulo,
                     filename):
    """Comparative plot (in addition to the six summary figures)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(spec_real['abs_eig'], 'o-', label='real',
                 markersize=4, color='C0')
    axes[0].plot(spec_cc['abs_eig'], 's-', label='classical Cramer',
                 markersize=4, alpha=0.7, color='C2')
    axes[0].plot(spec_cr['abs_eig'], '^-', label='refined Cramer',
                 markersize=4, alpha=0.7, color='C3')
    axes[0].plot(spec_shuf['abs_eig'], 'd-', label='shuffle',
                 markersize=4, alpha=0.7, color='C1')
    axes[0].set_xlabel('Eigenvalue index')
    axes[0].set_ylabel('|lambda|')
    axes[0].set_title(f'Spectrum - mod {modulo}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.cumsum(spec_real['contrib']) * 100, 'o-',
                 label='real', markersize=4, color='C0')
    axes[1].plot(np.cumsum(spec_cc['contrib']) * 100, 's-',
                 label='classical Cramer', markersize=4, alpha=0.7,
                 color='C2')
    axes[1].plot(np.cumsum(spec_cr['contrib']) * 100, '^-',
                 label='refined Cramer', markersize=4, alpha=0.7,
                 color='C3')
    axes[1].set_xlabel('Eigenvalue index')
    axes[1].set_ylabel('% cumulative contribution')
    axes[1].set_title(f'Cumulative contribution - mod {modulo}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(spec_real['pi'], 'o-', label='real',
                 markersize=4, color='C0')
    axes[2].plot(spec_cc['pi'], 's-', label='classical Cramer',
                 markersize=4, alpha=0.7, color='C2')
    axes[2].plot(spec_cr['pi'], '^-', label='refined Cramer',
                 markersize=4, alpha=0.7, color='C3')
    axes[2].axhline(1/len(spec_real['pi']), color='k', linestyle='--',
                    alpha=0.5, label='uniform')
    axes[2].set_xlabel('State')
    axes[2].set_ylabel('pi_i')
    axes[2].set_title(f'Stationary distribution - mod {modulo}')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time()
    print("=" * 80)
    print("  PRIME RESIDUE SPECTRAL ANALYSIS (Extended)")
    print("=" * 80)
    print(f"N_PRIMOS: {N_PRIMOS:,} | Moduli: {MODULOS}")
    print(f"Alpha Dirichlet: {ALPHA} | Seed: {SEMILLA}")
    print(f"Deep gap exploration + 10 narrow ranges + sliding windows")
    print()

    verifier = SelfVerifier()

    primes = generate_primes(N_PRIMOS, verifier=verifier)
    gaps = compute_gaps(primes, verifier=verifier)
    print()

    # Gap analysis with deep exploration
    print("\n" + "=" * 80)
    print("  GAP ANALYSIS")
    print("=" * 80)

    gap_values = sorted(set(gaps))
    gap_map = {g: i for i, g in enumerate(gap_values)}
    obs_gaps = np.array([gap_map[g] for g in gaps])
    K_g = len(gap_values)

    P_gaps, _ = transition_matrix_order1(obs_gaps, K_g)
    spec_gaps = analyze_spectrum(P_gaps, K_g)

    print(f"\nGap spectrum:")
    print(f"  K (unique gap values) = {K_g}")
    print(f"  |lambda2| = {spec_gaps['abs_eig'][1]:.5f}")
    print(f"  H/H_max = {spec_gaps['entropy_ratio']:.5f}")
    print(f"  Mixing time = {spec_gaps['t_mix']:.2f}")

    obs_gaps_shuf = shuffle_observations(obs_gaps, SEMILLA)
    P_gaps_shuf, _ = transition_matrix_order1(obs_gaps_shuf, K_g)
    spec_gaps_shuf = analyze_spectrum(P_gaps_shuf, K_g)

    rng = np.random.default_rng(SEMILLA)
    obs_gaps_cram = rng.choice(obs_gaps, size=len(obs_gaps), replace=True)
    P_gaps_cram, _ = transition_matrix_order1(obs_gaps_cram, K_g)
    spec_gaps_cram = analyze_spectrum(P_gaps_cram, K_g)

    diff_shuf = spec_gaps['abs_eig'][1] - spec_gaps_shuf['abs_eig'][1]
    diff_cram = spec_gaps['abs_eig'][1] - spec_gaps_cram['abs_eig'][1]

    print(f"\n  Comparison with null models:")
    print(f"    |lambda2| real    = {spec_gaps['abs_eig'][1]:.5f}")
    print(f"    |lambda2| shuffle = {spec_gaps_shuf['abs_eig'][1]:.5f}")
    print(f"    |lambda2| Cramer  = {spec_gaps_cram['abs_eig'][1]:.5f}")
    print(f"    Delta vs shuffle = {diff_shuf:+.5f}")
    print(f"    Delta vs Cramer  = {diff_cram:+.5f}")

    # Deep exploration
    print("\n" + "#" * 80)
    print("#  DEEP GAP EXPLORATION")
    print("#" * 80)

    blocks = gaps_by_blocks(gaps, verifier=verifier)
    significance = gaps_significance_test(gaps, verifier=verifier)

    primes_arr = np.array(primes)
    log_primes = np.log(primes_arr[:-1])
    normalized = normalized_gaps_test(primes, gaps, verifier=verifier)

    corr_210 = residue_gap_correlation(primes, 210, verifier=verifier)
    corr_2310 = residue_gap_correlation(primes, 2310, verifier=verifier)

    print("\n" + "#" * 80)
    print("#  ADDITIONAL TESTS")
    print("#" * 80)
    stat_gaps = stationarity_test_gaps(gaps, verifier=verifier)
    ks = ks_test_normalized_gaps(gaps, log_primes, verifier=verifier)

    print("\n" + "#" * 80)
    print("#  EXTENDED EXPLORATION (Option B)")
    print("#" * 80)
    narrow_results = gap_analysis_narrow_ranges(primes,
                                                 verifier=verifier)
    sliding_results = sliding_window_density_analysis(primes,
                                                       verifier=verifier)

    gap_info = {
        'spec_gaps': spec_gaps,
        'spec_gaps_shuf': spec_gaps_shuf,
        'spec_gaps_cram': spec_gaps_cram,
        'diff_gaps_shuf': diff_shuf,
        'diff_gaps_cram': diff_cram,
        'deep_analysis': {
            'blocks': blocks,
            'significance': significance,
            'normalized': normalized,
            'corr_210': corr_210,
            'corr_2310': corr_2310,
        },
        'stationarity_gaps': stat_gaps,
        'ks': ks,
    }

    # Residue analysis
    results_by_modulus = {}
    for modulo in MODULOS[:2]:
        print("\n" + "=" * 80)
        print(f"  MODULUS {modulo}")
        print("=" * 80)

        obs_real, coprimes = residues_to_indices(primes, modulo,
                                                   verifier=verifier)
        K = len(coprimes)
        print(f"Observations: {len(obs_real):,} | K = {K} symbols")

        obs_shuf = shuffle_observations(obs_real, SEMILLA)
        obs_cram_classical = generate_cramer_classical(
            len(obs_real), modulo, SEMILLA, verifier=verifier)
        obs_cram_ref = generate_cramer_refined(
            len(obs_real), modulo, gaps, SEMILLA)

        spec_real, spec_cc, spec_cr = compare_refined_cramer(
            obs_real, obs_cram_ref, obs_cram_classical, K,
            verifier=verifier)

        order2_info = order2_markov_test(obs_real, obs_shuf,
                                          obs_cram_classical, K)

        stat_res = stationarity_test_residues(
            obs_real, K, verifier=verifier,
            name=f"residues mod {modulo}")

        P_shuf, _ = transition_matrix_order1(obs_shuf, K)
        spec_shuf = analyze_spectrum(P_shuf, K)

        plot_comparison(spec_real, spec_cc, spec_cr, spec_shuf, modulo,
                         f'spectrum_mod{modulo}.png')

        comp = {
            'lambda2_real': spec_real['abs_eig'][1],
            'lambda2_shuf': spec_shuf['abs_eig'][1],
            'lambda2_cc': spec_cc['abs_eig'][1],
            'lambda2_cr': spec_cr['abs_eig'][1],
            'H_ratio_real': spec_real['entropy_ratio'],
            'H_ratio_shuf': spec_shuf['entropy_ratio'],
            'H_ratio_cc': spec_cc['entropy_ratio'],
            'H_ratio_cr': spec_cr['entropy_ratio'],
        }

        results_by_modulus[modulo] = {
            'K': K,
            'comparison': comp,
            'spec_real': spec_real,
            'spec_cc': spec_cc,
            'spec_cr': spec_cr,
            'spec_shuf': spec_shuf,
            'order2_info': order2_info,
            'stationarity_residues': stat_res,
        }

    # Large moduli
    print("\n" + "=" * 80)
    print("  LARGE MODULI")
    print("=" * 80)
    large_moduli = []
    for modulo in MODULOS[2:]:
        res = analyze_large_modulus(primes, modulo, verifier=verifier)
        if res:
            large_moduli.append(res)

    # Conclusion
    conclude(results_by_modulus, gap_info, large_moduli,
             narrow_results, sliding_results)

    # Self-verification
    verifier.summary()

    # ============================================================
    # GENERATE FIGURES
    # ============================================================
    print("\n" + "=" * 80)
    print("  GENERATING FIGURES")
    print("=" * 80)

    figure1_spectrum(results_by_modulus)
    figure2_gaps(gap_info, primes)
    figure3_blocks(gap_info)
    figure4_normalized(gap_info, narrow_results)
    figure5_sliding(sliding_results)
    figure6_summary(results_by_modulus, gap_info, narrow_results)

    print(f"\nTotal time: {time() - t0:.1f}s")


if __name__ == "__main__":
    main()
