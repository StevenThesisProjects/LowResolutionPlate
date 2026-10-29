#!/bin/sh
# Rebuild every diagram in figures/src/ to a vector PDF in figures/.
#   sh build_figures.sh            # all figures
#   sh build_figures.sh erd gantt  # only these
# Requires tectonic (brew install tectonic) or replace with pdflatex.
set -e
cd "$(dirname "$0")/figures/src"
names="$*"
[ -z "$names" ] && names=$(ls *.tex | grep -v '^figstyle.tex$' | sed 's/\.tex$//')
for n in $names; do
  tectonic --chatter minimal --outdir .. "$n.tex"
  echo "built figures/$n.pdf"
done
