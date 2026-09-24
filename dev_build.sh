#!/usr/bin/env bash
# Fast in-place development build: compiles the extension into src/shufflesnap/.
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python)" >/dev/null
cmake --build build
cp build/_core*.so src/shufflesnap/
