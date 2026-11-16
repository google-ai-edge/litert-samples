// swift-tools-version: 5.9
// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import PackageDescription

// The LiteRT-LM 0.19.0 runtime for macOS, pinned by artifact: the CLiteRTLM_mac.xcframework.zip asset of the
// LiteRT-LM v0.19.0 release, checked against its SHA-256 when SwiftPM downloads it. The app calls its C API
// (engine.h, conversation.h, model_info.h); see the README for why not the LiteRT-LM Swift package's tag.
let package = Package(
  name: "LiteRTLMRuntime",
  platforms: [.macOS(.v14)],
  products: [
    .library(name: "CLiteRTLM", targets: ["CLiteRTLM_mac"])
  ],
  targets: [
    .binaryTarget(
      name: "CLiteRTLM_mac",
      url:
        "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.19.0/CLiteRTLM_mac.xcframework.zip",
      checksum: "1a9e283a99a119aefe21afda541a5308a7d88aa9cb12f58aa7974038e3369c7c"
    )
  ]
)
