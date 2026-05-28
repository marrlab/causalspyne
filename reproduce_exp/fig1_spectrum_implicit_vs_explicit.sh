#!/usr/bin/env bash
# Figure: Power spectrum comparison — implicit vs explicit DGP
# Paper section: experiment_spectrum.tex (Table + fig:spectrum_comparison)
#
# Produces:
#   $OUTDIR/spectrum/spectrum_comparison.pdf
#   $OUTDIR/spectrum/spectrum_comparison.png
#   $OUTDIR/spectrum/implicit_eigvals.npy
#   $OUTDIR/spectrum/explicit_eigvals.npy
#
# Usage:
#   bash reproduce_exp/fig1_spectrum_implicit_vs_explicit.sh          # full (200 trials)
#   QUICK=1 bash reproduce_exp/fig1_spectrum_implicit_vs_explicit.sh  # quick test
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_ROOT"

OUT="$OUTDIR/spectrum"
LOG="/tmp/reproduce_fig1.log"
mkdir -p "$OUT"

echo "=== Fig 1: Implicit vs Explicit power spectrum ==="
echo "Trials: $SPECTRUM_TRIALS | Output: $OUT | Log: $LOG"

uv run python examples/compare_spectrum_implicit_vs_explicit.py \
  --trials "$SPECTRUM_TRIALS" \
  --num-macro-nodes 4 \
  --size-micro 3 \
  --degree 2.0 \
  --seed 42 \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

echo "Done. Key outputs:"
echo "  $OUT/spectrum_comparison.pdf"
echo "  $OUT/spectrum_comparison.png"
