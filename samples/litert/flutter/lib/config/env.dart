// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Compile-time configuration. Pass values with `--dart-define=KEY=value` or
/// `--dart-define-from-file=.env` (gitignored). Anything here is compiled into
/// the binary.
library;

/// A local Gemma `.litertlm` file: an absolute path, or a path relative to
/// the app's documents directory. Empty means "the chat model chosen in the
/// Chat model card" (or none). Set, it fills the chat slot while no model is
/// chosen there, with Gemma 4 E2B's settings (`kDefineChatModel`).
const kGemmaModelPath = String.fromEnvironment('GEMMA_MODEL_PATH');

/// `gpu` or `cpu`; empty (the default) leaves it to the Live camera's
/// Detector setting (GPU unless the user chose CPU). Set, it wins over the
/// setting, which the Live camera then shows locked. `cpu` is an explicit,
/// labelled mode, never a fallback. Validated at the detector's load.
const kDetectorBackend = String.fromEnvironment('DETECTOR_BACKEND');

/// Where the Live camera's frames come from: `camera` (default) is the
/// user's choice in its settings (the device camera, or a network camera);
/// `fixture` and `network` are fixed by the build and lock that choice.
const kFrameSource = String.fromEnvironment(
  'FRAME_SOURCE',
  defaultValue: 'camera',
);

/// The MJPEG URL for `FRAME_SOURCE=network`, e.g.
/// `http://192.168.1.23:8080/video` (a phone running IP Webcam).
const kNetworkCameraUrl = String.fromEnvironment('NETWORK_CAMERA_URL');

/// Directory of still images for `FRAME_SOURCE=fixture`.
const kFixtureDir = String.fromEnvironment('FIXTURE_DIR');

/// Silence gate in dBFS (default −45): a 20 ms frame must be at least this
/// loud to count as speech. Tune for a noisy room; validated at startup.
const kVoiceGateDbfs = String.fromEnvironment('VOICE_GATE_DBFS');
