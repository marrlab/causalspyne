#!/usr/bin/env bash
# Master script: reproduce all paper figures in sequence.
#
# Usage:
#   bash reproduce_exp/reproduce_all.sh           # full paper results (~4-8h)
#   QUICK=1 bash reproduce_exp/reproduce_all.sh   # quick smoke-test (~10 min)
#   OUTDIR=my_output bash reproduce_exp/reproduce_all.sh
#
# Individual figures can be run independently:
#   bash reproduce_exp/fig1_spectrum_implicit_vs_explicit.sh
#   bash reproduce_exp/fig2_fci_root_vs_intermediate.sh
#   bash reproduce_exp/fig3_multilevel_hidden.sh
#   bash reproduce_exp/fig4_spectrum_vs_fci_shd.sh
#   bash reproduce_exp/fig5_uml_architecture.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/config.sh"

START=$(date +%s)
echo "======================================================"
echo " CausalSpyne — Reproduce All Paper Figures"
echo " Mode:    $([ "$QUICK" = "1" ] && echo QUICK || echo FULL)"
echo " Output:  $OUTDIR"
echo " Started: $(date)"
echo "======================================================"

run_fig() {
  local script="$1"
  local name="$2"
  echo ""
  echo "------------------------------------------------------"
  echo " Running: $name"
  echo "------------------------------------------------------"
  bash "$DIR/$script"
}

run_fig fig1_spectrum_implicit_vs_explicit.sh "Fig 1: Implicit vs Explicit power spectrum"
run_fig fig2_fci_root_vs_intermediate.sh      "Fig 2: FCI root vs intermediate (paired)"
run_fig fig3_multilevel_hidden.sh             "Fig 3: Multilevel hidden confounder (5 positions)"
run_fig fig4_spectrum_vs_fci_shd.sh          "Fig 4: Spectral range vs FCI SHD"
run_fig fig5_uml_architecture.sh             "Fig 5: UML architecture diagrams"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "======================================================"
echo " All figures complete in ${ELAPSED}s"
echo " Output directory: $REPO_ROOT/$OUTDIR"
echo "======================================================"

echo ""
echo "Summary of produced files:"
find "$REPO_ROOT/$OUTDIR" -name "*.pdf" -o -name "*.png" 2>/dev/null | sort
