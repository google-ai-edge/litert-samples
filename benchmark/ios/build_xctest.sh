#!/bin/bash
# Builds the app and its XCTest bundle for a device lab and zips them the way Developer Device
# Platform takes an iOS test (https://docs.cloud.google.com/developer-device-platform/device-run/ios):
# the products directory and the .xctestrun manifest at the root of the archive, plus BUILD_INFO
# (the framework's build, which run_ddp_ios.py writes into each session). The test action builds
# Release, so the archive holds the build the rows come from. The lab re-signs the app, but
# xcodebuild signs it once here.
#
# Usage: DEVELOPMENT_TEAM=<Apple team id> [BUNDLE_ID=<app bundle id>] [DEVELOPER_DIR=<Xcode.app>] \
#          ./build_xctest.sh [<out.zip>]
# Needs Frameworks/ from build_framework.sh and Xcode; DEVELOPER_DIR picks the Xcode when several
# are installed (the lab runs Xcode 26.2 by default). Default output: out/LiteRTBenchmark-xctest.zip.
# BUNDLE_ID (default com.example.LiteRTBenchmark) is the app's id inside the archive; the test
# bundle takes the same id with ".tests" appended, and run_ddp_ios.py reads the app's id back
# from the archive's manifest.
# Exit status: 1 without DEVELOPMENT_TEAM or when xcodebuild or zip fail, 2 without the framework
# or the build info, 3 when xcodebuild wrote no manifest or no Release products.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM to your Apple team id (xcodebuild signs the app before the lab re-signs it)}"
OUT="${1:-$HERE/out/LiteRTBenchmark-xctest.zip}"
case "$OUT" in *.zip) ;; *) OUT="$OUT.zip";; esac  # zip appends the suffix itself otherwise
BUILD="$HERE/out/build"
for needed in "$HERE/Frameworks/LiteRtBenchmark.framework" "$HERE/Frameworks/BUILD_INFO"; do
  if [ ! -e "$needed" ]; then
    echo "no $needed: run ./build_framework.sh first" >&2
    exit 2
  fi
done
rm -rf "$BUILD"
xcodebuild build-for-testing -project "$HERE/LiteRTBenchmark.xcodeproj" -scheme LiteRTBenchmark \
  -destination "generic/platform=iOS" -derivedDataPath "$BUILD" -allowProvisioningUpdates \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" ${BUNDLE_ID:+LITERT_BENCHMARK_BUNDLE_ID="$BUNDLE_ID"} -quiet
PRODUCTS="$BUILD/Build/Products"
if ! ls "$PRODUCTS"/*.xctestrun >/dev/null 2>&1 || [ ! -d "$PRODUCTS/Release-iphoneos" ]; then
  echo "xcodebuild left no .xctestrun or no Release-iphoneos under $PRODUCTS; nothing to submit" >&2
  exit 3
fi
mkdir -p "$(dirname "$OUT")"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"  # zip runs from the products directory
rm -f "$OUT"
cp "$HERE/Frameworks/BUILD_INFO" "$PRODUCTS/BUILD_INFO"
(cd "$PRODUCTS" && zip -qrMM "$OUT" Release-iphoneos ./*.xctestrun BUILD_INFO -x "*.dSYM/*")
echo "test archive: $OUT ($(du -h "$OUT" | cut -f1))"
