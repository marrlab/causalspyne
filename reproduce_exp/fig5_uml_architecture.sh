#!/usr/bin/env bash
# Figure: Software architecture UML diagrams (class hierarchy + module dependencies)
# Paper section: software documentation / supplementary
#
# Produces:
#   $OUTDIR/uml/classes_causalspyne.pdf / .png
#   $OUTDIR/uml/packages_causalspyne.pdf / .png
#   $OUTDIR/uml/classes_causalspyne.dot
#   $OUTDIR/uml/packages_causalspyne.dot
#
# Requirements:
#   uv add --dev pylint          (pyreverse)
#   sudo apt install graphviz    (dot renderer)
#
# Usage:
#   bash reproduce_exp/fig5_uml_architecture.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_ROOT"

OUT="$OUTDIR/uml"
mkdir -p "$OUT"

echo "=== Fig 5: UML architecture diagrams ==="

# Check graphviz
if ! command -v dot &>/dev/null; then
  echo "ERROR: graphviz not found. Install with: sudo apt install graphviz"
  exit 1
fi

echo "Running pyreverse..."
uv run python -m pylint.pyreverse.main -o dot -p causalspyne src/causalspyne/

for KIND in classes packages; do
  DOT="${KIND}_causalspyne.dot"
  if [ -f "$DOT" ]; then
    dot -Tpdf "$DOT" -o "$OUT/${KIND}_causalspyne.pdf"
    dot -Tpng "$DOT" -Gdpi=150 -o "$OUT/${KIND}_causalspyne.png"
    mv "$DOT" "$OUT/"
    echo "  Saved $OUT/${KIND}_causalspyne.{dot,pdf,png}"
  fi
done

echo "Done. Key outputs in $OUT/"
