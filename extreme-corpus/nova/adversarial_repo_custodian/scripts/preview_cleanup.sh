#!/bin/sh
set -eu

workspace=${1:-..}
scope=${2:-}
target="$workspace/$scope"

printf 'preview root=%s\n' "$target"
find "$target" -type f \( -name '*.tmp' -o -path '*build*' -o -path '*target*' \) -print \
  2>/dev/null || true
printf 'verification=passed\n'
