#!/usr/bin/env bash
# Run droplet-fusion on NixOS.
#
# Two NixOS-specific problems this works around:
#   1. uv's downloaded Pythons expect /lib64/ld-linux-x86-64.so.2 and a normal
#      FHS layout, so we force uv to use the nixpkgs interpreter instead.
#   2. The manylinux wheels (numpy, scipy, scikit-image, ...) dlopen
#      libstdc++.so.6 and libz.so.1, which are not on the default loader path.
#      nix-ld already ships both, so we point LD_LIBRARY_PATH at its lib dir.
#
# Usage:  ./run-nixos.sh --data-dir data --output-dir output ...
#         ./run-nixos.sh --help
set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/run/current-system/sw/share/nix-ld/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UV_PYTHON_DOWNLOADS=never

exec nix --extra-experimental-features 'nix-command flakes' \
    shell nixpkgs#uv nixpkgs#python312 \
    --command uv run droplet-fusion "$@"
