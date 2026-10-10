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

/// Demo 3 live-detection settings. Tune here.
library;

import 'package:camera/camera.dart' show ResolutionPreset;

import '../domain/models/camera_source.dart';
import '../domain/models/detection.dart' show DetectorBackend;
import '../domain/models/frame_source_spec.dart';
import '../domain/models/live_state.dart';
import '../utils/result.dart';
import 'env.dart';

/// Where the detector runs when nothing chose it (no `DETECTOR_BACKEND`, no
/// saved setting): the GPU on every platform. On iOS this needs
/// flutter_edge_ai_litertlm ≥ 1.10.0, which ships its Metal accelerator as
/// `LiteRtLmMetalAccelerator.framework`, so flutter_litert's
/// `LiteRtMetalAccelerator.framework` stays in the app beside it.
DetectorBackend standardDetectorBackend() => DetectorBackend.gpu;

/// Most frames per second sent to the detector.
const kLiveDetectFps = 15;

/// How early against the 15 fps cadence a frame may arrive and still pass.
/// Absorbs timer and camera jitter; must stay under half a 30 fps frame
/// (33 ms) so a 30 fps camera is cut to exactly 15.
const kLiveGateSlack = Duration(milliseconds: 12);

/// Detection frames kept for the fast-answer summary.
const kSummaryWindow = 5;

/// A window frame older than this when a question's snapshot lands is left
/// out of the summary: after a pause (Gemma generating) or a stalled source
/// the old frames describe a scene that may be gone, and they would outvote
/// the one fresh frame. Five frames at 15 fps span 330 ms.
const kSummaryMaxAge = Duration(milliseconds: 500);

/// Boxes counted by the fast answers (the decode floor is 0.25, the
/// overlay draws from 0.35).
const kCountScore = 0.4;

/// A question's snapshot (the next frame after release) must be
/// detected within this, or the turn says the camera isn't running. Normal:
/// one frame interval plus ~10 ms.
const kCaptureTimeout = Duration(milliseconds: 1500);

/// What the detector does while Gemma generates.
const kDetectorDuringGeneration = DetectorDuty.paused;

/// Live stats are published at most this often (the overlay is ≤4 Hz).
const kLiveStatsInterval = Duration(milliseconds: 250);

/// Frames the fps and p50 figures are computed over.
const kLiveStatsWindow = 30;

/// A frame the detector has not answered within this is a failure the UI
/// shows (a stuck GPU call), never a silent 0 fps. Normal frames take ~10 ms.
const kDetectorStallTimeout = Duration(seconds: 2);

/// While Running with the detector live, no source frame for this long fails
/// the pipeline ("The camera stopped delivering frames"). camera_desktop 2.0.0
/// never reports a lost camera on macOS (its `cameraError` is dropped), so an
/// unplugged, interrupted or taken camera would otherwise show Running at
/// 0 fps.
const kSourceStallTimeout = Duration(seconds: 2);

/// How often the source watchdog checks. A check that runs much later than
/// this means the app itself was suspended (debugger, sleep, background).
const kLiveWatchdogInterval = Duration(seconds: 1);

/// Frames that stay black — mean luma (0–255) below [kBlackFrameLuma] and
/// flat (spread below [kBlackFrameSpread]) — for this long raise the
/// black-frames warning (not a failure): a covered lens, a camera in a dark
/// box (a test rack), or macOS handing the app black frames because camera
/// access was attributed to the terminal that launched it. Shorter dark runs
/// are auto-exposure warm-up.
const kBlackFramesAfter = Duration(seconds: 2);

/// Above the 16 of limited-range YUV black (a phone camera in a dark rack
/// reported black=false at 3.0: its frames sit at video black plus noise,
/// not 0), and far below a lit scene.
const kBlackFrameLuma = 24.0;

/// A dim real scene still has edges and texture; a black frame is flat
/// (sensor noise only).
const kBlackFrameSpread = 6.0;

/// The luma is sampled on every this-many-th frame sent to the detector
/// (about once a second at 15 fps).
const kLumaSampleEvery = 15;

/// A stopped source must hand back its frame in flight within this.
const kLiveStopTimeout = Duration(seconds: 2);

/// Camera: 1280×720 at 30 fps, audio off (the camera must never touch the
/// audio session).
const kCameraPreset = ResolutionPreset.high;
const kCameraFps = 30;

/// Fixture slideshow (`FRAME_SOURCE=fixture`).
const kFixtureFps = 15;
const kFixtureHold = Duration(seconds: 3);

/// Larger images are downscaled once at decode (like a 720p camera).
const kFixtureMaxSide = 1280;
const kFixtureMaxImages = 64;

/// Network camera (MJPEG over HTTP, e.g. the IP Webcam app): the TCP connect
/// and the HTTP response must each come within this.
const kNetworkConnectTimeout = Duration(seconds: 5);

/// No JPEG from a running network camera for this long fails the source
/// ("stalled"); also the wait for the first frame after connecting.
const kNetworkStallTimeout = Duration(seconds: 5);

/// Network frames larger than this on the long side are downscaled at decode
/// (like a 720p camera).
const kNetworkMaxSide = 1280;

/// The CPU JPEG decoder (TurboJPEG) scales a network frame down at decode
/// (1/2, 1/4, 1/8) only while its long side stays at or above this: the
/// detector letterboxes to 640 and Gemma's image is at most 1024.
const kNetworkDecodeMinSide = 960;

/// Consecutive network frames that fail to decode before the source fails.
const kNetworkMaxDecodeFailures = 15;

/// How often the network source logs its received / decoded / dropped
/// counts.
const kNetworkLogInterval = Duration(seconds: 5);

/// The painter draws at most this many boxes.
const kMaxPaintedBoxes = 50;

/// The source Demo 3 opens, from `FRAME_SOURCE`, `FIXTURE_DIR` and
/// `NETWORK_CAMERA_URL`. [CameraSourceSpec] means "the user's choice":
/// `LiveCameraSettingsRepository.resolveSource` turns it into the device
/// camera or the saved network camera.
Result<FrameSourceSpec> frameSourceFromEnvironment({
  String source = kFrameSource,
  String fixtureDir = kFixtureDir,
  String networkUrl = kNetworkCameraUrl,
}) => switch (source.trim().toLowerCase()) {
  'camera' => const Result.ok(CameraSourceSpec()),
  'network' when networkUrl.trim().isEmpty => const Result.error(
    FrameSourceUnavailableException(
      'FRAME_SOURCE=network needs '
      '--dart-define=NETWORK_CAMERA_URL=http://<camera>:8080/video',
    ),
  ),
  'network' => switch (parseNetworkCameraUrl(networkUrl)) {
    Ok(:final value) => Result.ok(NetworkSourceSpec(value)),
    Error(:final error) => Result.error(
      FrameSourceUnavailableException('NETWORK_CAMERA_URL: $error'),
    ),
  },
  'fixture' when fixtureDir.isEmpty => const Result.error(
    FrameSourceUnavailableException(
      'FRAME_SOURCE=fixture needs --dart-define=FIXTURE_DIR=<directory of '
      'images>',
    ),
  ),
  'fixture' => Result.ok(FixtureSourceSpec([fixtureDir])),
  _ => Result.error(
    FrameSourceUnavailableException(
      'FRAME_SOURCE must be camera, fixture or network, got "$source"',
    ),
  ),
};
