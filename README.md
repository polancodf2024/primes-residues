# prime_residue_spectral.py (v7)

Spectral analysis of Markov chains on prime residues, with deep gap exploration and self-verification.

## Overview

This program performs a systematic spectral investigation of the Markov chain structure of prime numbers. It analyzes prime residues modulo 210, 2310, and 30030, along with the gaps between consecutive primes, and compares them against three null models: shuffled sequences, classical Cramér pseudo-primes, and refined Cramér pseudo-primes that use the observed gap distribution.

The central finding is that prime residues exhibit strong sequential structure that is fully reproduced by the density $1/\log x$ plus the observed gap distribution. No evidence of deep arithmetic modular structure is found. The apparent structure in the prime gaps is contained in the marginal gap distribution, not in the sequential order.

## Requirements

- Python 3.10 or higher
- NumPy
- SciPy (for statistical tests)
- SymPy (for prime generation)
- Matplotlib (for figure generation)

Install dependencies:

```bash
pip install numpy scipy sympy matplotlib
