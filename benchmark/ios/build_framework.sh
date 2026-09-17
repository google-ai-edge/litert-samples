#!/bin/bash
# Builds LiteRtBenchmark.framework (LiteRT's benchmark_model for iOS arm64) from a LiteRT checkout at the pinned tag,
# fetches the matching prebuilt Metal accelerator, and puts both under Frameworks/ next to this script.
#
# Usage: [LITERT_SRC=/path/to/litert] ./build_framework.sh [tag]      (default tag: v2.2.0)
# Without LITERT_SRC the tag is cloned into ./litert-src. With it, that checkout is switched to the tag
# (detached HEAD) and gains litert/tools/ios_benchmark/ (the three files in ./bazel); configure.py runs
# once if the checkout has no .litert_configure.bazelrc.
# Needs: bazelisk, Xcode command line tools, python3 (for LiteRT's configure.py), curl.
set -euo pipefail
TAG="${1:-v2.2.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LITERT_SRC="${LITERT_SRC:-$HERE/litert-src}"
if [ ! -e "$LITERT_SRC/.git" ]; then  # a worktree has a .git file, a clone a .git directory
  git clone --depth 1 --branch "$TAG" https://github.com/google-ai-edge/litert.git "$LITERT_SRC"
fi
git -C "$LITERT_SRC" checkout -q "$TAG"
# The framework target lives in this directory, not in LiteRT: copy it into the checkout.
mkdir -p "$LITERT_SRC/litert/tools/ios_benchmark"
cp "$HERE/bazel/BUILD" "$HERE/bazel/litert_benchmark_shim.cc" "$HERE/bazel/litert_benchmark_shim.h" "$LITERT_SRC/litert/tools/ios_benchmark/"
cd "$LITERT_SRC"
[ -f .litert_configure.bazelrc ] || PYTHON_BIN_PATH="$(command -v python3)" TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_CONFIGURE_IOS=0 python3 configure.py
# BENCHMARK_MODEL_NO_MAIN drops the tool's own main() (see litert/tools/benchmark_litert_model_main.cc); the app calls Main via the shim.
bazelisk build --config=ios_arm64 -c opt --copt=-DBENCHMARK_MODEL_NO_MAIN //litert/tools/ios_benchmark:LiteRtBenchmark_framework
rm -rf "$HERE/Frameworks/LiteRtBenchmark.framework"
mkdir -p "$HERE/Frameworks"
unzip -o -q bazel-bin/litert/tools/ios_benchmark/LiteRtBenchmark_framework.zip -d "$HERE/Frameworks"
VERSION="${TAG#v}"
curl -sSL -o "$HERE/Frameworks/libLiteRtMetalAccelerator.dylib" \
  "https://storage.googleapis.com/litert/binaries/$VERSION/ios_arm64/libLiteRtMetalAccelerator.dylib"
echo "framework: $HERE/Frameworks/LiteRtBenchmark.framework"
echo "accelerator: $HERE/Frameworks/libLiteRtMetalAccelerator.dylib"
