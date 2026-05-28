#!/usr/bin/env bash
# Figure: Spectral range of observed precision vs FCI SHD
# Tests whether instances with wider spectral range are easier for FCI.
# Paper section: analysis_spectrum_ci_difficulty.tex (empirical validation)
#
# Produces:
#   $OUTDIR/spectrum_vs_shd/spectrum_range_vs_shd.pdf / .png
#   $OUTDIR/spectrum_vs_shd/results.csv
#
# Usage:
#   bash reproduce_exp/fig4_spectrum_vs_fci_shd.sh
#   QUICK=1 bash reproduce_exp/fig4_spectrum_vs_fci_shd.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_ROOT"

OUT="$OUTDIR/spectrum_vs_shd"
LOG="/tmp/reproduce_fig4.log"
mkdir -p "$OUT"

echo "=== Fig 4: Spectral range vs FCI SHD ==="
echo "Seeds/combo: $SEEDS_PER_COMBO | Samples: $SAMPLES | Output: $OUT | Log: $LOG"

uv run python examples/compare_spectrum_radius_vs_shd.py \
  --macro-node-sizes 3 4 \
  --degrees 1.5 2.0 3.0 \
  --seeds-per-combo "$SEEDS_PER_COMBO" \
  --samples "$SAMPLES" \
  --min-num-local-nodes "$MIN_MICRO" \
  --max-num-local-nodes "$MAX_MICRO" \
  --alpha "$ALPHA" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

echo "Done. Key outputs:"
echo "  $OUT/spectrum_range_vs_shd.pdf"
echo "  $OUT/spectrum_range_vs_shd.png"
