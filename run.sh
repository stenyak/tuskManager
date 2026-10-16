#!/usr/bin/env bash
# Builds tuskManager if its sources changed, then launches it.
set -e
cd "$(dirname "$0")"

# The binary is copied here from the build files.
export CARGO_TARGET_DIR="$PWD/target"

# rustup adds cargo to PATH, but shells opened before the install don't see it yet.
export PATH="$HOME/.cargo/bin:$PATH"

missing=
need() { command -v "$1" >/dev/null 2>&1 || missing+=" - '$1' command not found: install $2"$'\n'; }
need cc "a C compiler and linker, such as gcc"
need cargo "a recent Rust toolchain, such as via rustup"
if [ -n "$missing" ]; then
    printf '\nYou need to install some tools:\n%s - Then press enter and rerun %s again\n\n' "$missing" "$0"
    read -r
    exit 1
fi

echo "Building tuskManager (the first build takes a few minutes)..."
cargo build --release
# -f replaces the binary even while it's running.
cp -f "$CARGO_TARGET_DIR/release/tuskManager" tuskManager
exec ./tuskManager "$@"
