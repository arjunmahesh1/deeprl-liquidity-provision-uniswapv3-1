#!/usr/bin/env bash
# Compile the memo in a temp dir and copy only the finished PDF up to reports/.
# Keeps .aux/.log/.fls clutter out of the working tree.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(dirname "$HERE")"
SRC="ICMLA_revision_memo.tex"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

latexmk -pdf -interaction=nonstopmode -halt-on-error \
        -outdir="$BUILD" "$HERE/$SRC" >/dev/null 2>&1 || {
  echo "compile FAILED; log follows:" >&2
  sed -n '/^!/,+6p' "$BUILD/${SRC%.tex}.log" >&2
  exit 1
}

echo "--- overfull boxes ---"
grep -E "Overfull \\\\[hv]box" "$BUILD/${SRC%.tex}.log" || echo "none"

cp "$BUILD/${SRC%.tex}.pdf" "$OUT/"
echo "--- wrote $OUT/${SRC%.tex}.pdf ---"
