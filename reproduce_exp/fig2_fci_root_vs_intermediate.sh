#!/usr/bin/env bash
# Figure: FCI SHD — root-hidden vs intermediate-hidden (paired, same DAG)
# Paper section: experiment_fci_benchmark.tex (Table + grouped boxplot)
#
# Produces:
#   $OUTDIR/fci_root_vs_intermediate/paired_shd_grouped_macro*.pdf
#   $OUTDIR/fci_root_vs_intermediate/paired_shd_grouped_macro*.png
#   $OUTDIR/fci_root_vs_intermediate/results.csv
#
# Usage:
#   bash reproduce_exp/fig2_fci_root_vs_intermediate.sh
#   QUICK=1 bash reproduce_exp/fig2_fci_root_vs_intermediate.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_ROOT"

OUT="$OUTDIR/fci_root_vs_intermediate"
LOG="/tmp/reproduce_fig2.log"
mkdir -p "$OUT"

echo "=== Fig 2: FCI root vs intermediate (paired, same DAG) ==="
echo "Replicates: $REPLICATES | Samples: $SAMPLES | Output: $OUT | Log: $LOG"

# Run each macro size separately for speed; results are merged in the output dir.
for MACRO in 3 4; do
  echo "--- macro=$MACRO ---"
  uv run python examples/compare_root_vs_intermediate_grouped.py \
    --macro-node-sizes "$MACRO" \
    --replicates "$REPLICATES" \
    --samples "$SAMPLES" \
    --min-num-local-nodes "$MIN_MICRO" \
    --max-num-local-nodes "$MAX_MICRO" \
    --alpha "$ALPHA" \
    --output-dir "$OUT" \
    2>&1 | tee -a "$LOG"
done

echo "Done. Key outputs in $OUT/"
