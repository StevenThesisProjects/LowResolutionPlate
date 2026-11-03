#!/bin/sh
# Rebuild every diagram in figures/src/ to a 300-dpi PNG in figures/.
#   sh build_figures.sh            # all figures
#   sh build_figures.sh erd gantt  # only these
# Requires tectonic (brew install tectonic) and pdftoppm (brew install poppler).
# The PNG stores its 300 dpi, so \includegraphics places it at the
# diagram's natural size (same 8 pt / 7 pt text in every figure).
set -e
cd "$(dirname "$0")/figures/src"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
names="$*"
[ -z "$names" ] && names=$(ls *.tex | grep -v '^figstyle.tex$' | sed 's/\.tex$//')
for n in $names; do
  tectonic --chatter minimal --outdir "$tmp" "$n.tex"
  pdftoppm -png -r 300 -singlefile "$tmp/$n.pdf" "../$n"
  echo "built figures/$n.png"
done
