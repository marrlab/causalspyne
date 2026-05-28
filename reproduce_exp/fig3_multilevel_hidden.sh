#!/usr/bin/env bash
# Figure: FCI SHD across 5 topological positions + power spectrum per position
# Paper section: multilevel hidden confounder analysis
#
# Produces:
#   $OUTDIR/multilevel/multilevel_shd_boxplot.pdf / .png
#   $OUTDIR/multilevel/multilevel_spectrum.pdf / .png
#   $OUTDIR/multilevel/results.csv
#
# Usage:
#   bash reproduce_exp/fig3_multilevel_hidden.sh
#   QUICK=1 bash reproduce_exp/fig3_multilevel_hidden.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_ROOT"

OUT="$OUTDIR/multilevel"
LOG="/tmp/reproduce_fig3.log"
mkdir -p "$OUT"

echo "=== Fig 3: Multilevel hidden confounder (5 positions) ==="
echo "Replicates: $REPLICATES | Samples: $SAMPLES | Output: $OUT | Log: $LOG"

uv run python examples/compare_multilevel_hidden.py \
  --macro-node-sizes 4 \
  --replicates "$REPLICATES" \
  --samples "$SAMPLES" \
  --min-num-local-nodes "$MIN_MICRO" \
  --max-num-local-nodes "$MAX_MICRO" \
  --alpha "$ALPHA" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

echo "Done. Key outputs:"
echo "  $OUT/multilevel_shd_boxplot.pdf"
echo "  $OUT/multilevel_spectrum.pdf"
