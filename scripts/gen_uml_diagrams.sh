#!/usr/bin/env bash
# Reproducible UML/architecture diagram generation for causalspyne.
# Requirements: pylint (pyreverse), graphviz (dot)
#   uv add --dev pylint
#   sudo apt install graphviz  (or brew install graphviz)
#
# Usage (from repo root):
#   bash scripts/gen_uml_diagrams.sh [output_dir]

set -euo pipefail
OUTDIR="${1:-output/uml}"
mkdir -p "$OUTDIR"

echo "Generating dot files via pyreverse..."
uv run python -m pylint.pyreverse.main \
    -o dot \
    -p causalspyne \
    src/causalspyne/

for kind in classes packages; do
    DOT="${kind}_causalspyne.dot"
    if [ -f "$DOT" ]; then
        dot -Tpdf "$DOT" -o "$OUTDIR/${kind}_causalspyne.pdf"
        dot -Tpng "$DOT" -Gdpi=150 -o "$OUTDIR/${kind}_causalspyne.png"
        mv "$DOT" "$OUTDIR/"
        echo "Saved $OUTDIR/${kind}_causalspyne.{dot,pdf,png}"
    fi
done

echo "Done. Files in $OUTDIR/"
