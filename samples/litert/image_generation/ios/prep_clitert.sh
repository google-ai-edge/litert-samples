#!/bin/bash
# Obtains CLiteRT.xcframework — the official Swift-facing LiteRT C API
# framework, bazel-built from the LiteRT repo (litert/swift:CLiteRT). The
# framework bundles the C headers, so no LiteRT code lives in this repo.
#
# Resolution order:
#   1. CLITERT_XCFRAMEWORK=<path to CLiteRT.xcframework> — copy it.
#   2. LITERT_CHECKOUT=<path> with an existing bazel-built
#      bazel-bin/litert/swift/CLiteRT.xcframework.zip — unzip it.
#   3. Fresh: clone the LiteRT repo into ./litert-src and bazel-build the
#      target (needs bazelisk + Xcode; ~10 min cold). Two stub BUILD files
#      are created for internal-only deps absent in the OSS tree.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d CLiteRT.xcframework ]; then
  echo "CLiteRT.xcframework already present"
  exit 0
fi

if [ -n "${CLITERT_XCFRAMEWORK:-}" ]; then
  cp -R "$CLITERT_XCFRAMEWORK" CLiteRT.xcframework
  echo "copied from $CLITERT_XCFRAMEWORK"
  exit 0
fi

CHECKOUT="${LITERT_CHECKOUT:-litert-src}"
ZIP="$CHECKOUT/bazel-bin/litert/swift/CLiteRT.xcframework.zip"

if [ ! -f "$ZIP" ]; then
  if [ ! -d "$CHECKOUT" ]; then
    echo "Cloning LiteRT into $CHECKOUT"
    git clone --depth 1 https://github.com/google-ai-edge/LiteRT.git "$CHECKOUT"
  fi
  # Internal-only implicit deps of ios_framework, absent in the OSS tree.
  if [ ! -f "$CHECKOUT/devtools/compliance/licenses/BUILD" ]; then
    mkdir -p "$CHECKOUT/devtools/compliance/licenses"
    cat > "$CHECKOUT/devtools/compliance/licenses/BUILD" << 'EOF'
# Local shim: internal-only implicit dep of ios_framework, absent in OSS.
filegroup(
    name = "no_external_contributions",
    visibility = ["//visibility:public"],
)
EOF
  fi
  if [ ! -f "$CHECKOUT/tools/cc_target_os/BUILD" ]; then
    mkdir -p "$CHECKOUT/tools/cc_target_os"
    cat > "$CHECKOUT/tools/cc_target_os/BUILD" << 'EOF'
# Local shim: internal-only select() keys, absent in OSS. Never true here.
config_setting(
    name = "emscripten",
    values = {"cpu": "wasm32-never-matches"},
    visibility = ["//visibility:public"],
)
EOF
  fi
  echo "Building litert/swift:CLiteRT (this can take ~10 min cold)"
  (cd "$CHECKOUT" && bazelisk build -c opt litert/swift:CLiteRT)
fi

unzip -q "$ZIP" -d .
echo "CLiteRT.xcframework ready:"
ls CLiteRT.xcframework
