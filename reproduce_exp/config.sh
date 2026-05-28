#!/usr/bin/env bash
# Shared configuration for all reproduce_exp scripts.
# Source this file at the top of each figure script:
#   source "$(dirname "$0")/config.sh"

# --- Output root (override with OUTDIR env var) ---
OUTDIR="${OUTDIR:-output/paper_figures}"

# --- Quick mode: fewer seeds/trials for local testing ---
# Set QUICK=1 for a fast dry-run; leave unset for full paper results.
QUICK="${QUICK:-0}"

# --- FCI significance threshold ---
ALPHA=0.05

# --- Micro-node range (must guarantee >=1 confounder per cluster) ---
MIN_MICRO=3
MAX_MICRO=5

# --- Derived: seeds and samples depending on mode ---
if [ "$QUICK" = "1" ]; then
  REPLICATES=5
  SAMPLES=100
  SEEDS_PER_COMBO=3
  SPECTRUM_TRIALS=20
  echo "[config] QUICK mode: reduced replicates/seeds for testing"
else
  REPLICATES=100       # paper: 100 seeds for FCI benchmarks
  SAMPLES=500          # paper: 500 samples
  SEEDS_PER_COMBO=20   # paper: 20 seeds per (macro, degree) combo
  SPECTRUM_TRIALS=200  # paper: 200 instances for spectrum comparison
fi

# --- Repo root (scripts must be run from repo root, or this adjusts) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
