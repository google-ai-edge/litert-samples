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
/// one frame interval plus the detection.
const kCaptureTimeout = Duration(milliseconds: 1500);

/// What the detector does while Gemma generates.
const kDetectorDuringGeneration = DetectorDuty.paused;

/// Live stats are published at most this often (the overlay is ≤4 Hz).
const kLiveStatsInterval = Duration(milliseconds: 250);

/// Frames the fps and p50 figures are computed over.
const kLiveStatsWindow = 30;

/// A frame the detector has not answered within this is a failure the UI
/// shows (a stuck GPU call), never a silent 0 fps. A normal frame takes a
/// small fraction of it.
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

/// Above the 16 of limited-range YUV black (a camera in the dark delivers
/// video black plus noise, not 0), and far below a lit scene.
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

// Visual memory: frames EmbeddingGemma 2 embeds while the camera demo is idle,
// and a spoken question searches by meaning.

/// The sampler asks for a memory frame no sooner than this after the end of
/// its previous attempt…
const kMemoryMinInterval = Duration(seconds: 3);

/// …nor sooner than this many times the last frame embedding took (a phone
/// that needs 1 s per frame embeds at most every 4 s).
const kMemoryEmbedSpacing = 4;

/// Once this much time has passed since the last stored or refreshed frame,
/// an unchanged scene refreshes the newest frame's time in place (a static
/// scene stays "seen just now" without a duplicate frame;
/// `VisualMemoryRepository`).
const kMemoryMaxInterval = Duration(seconds: 10);

/// How often the sampler checks whether a frame is due (its timer).
const kMemoryTick = Duration(milliseconds: 500);

/// A scene change, between two grey thumbnails of 32×32 cells (on 0–255)
/// once the reference is re-aligned for a hand-held camera's shift
/// ([kMemoryMaxShift]) and the overall brightness is matched (`sceneChange`
/// in `data/services/detector/memory_shot.dart`): a mean absolute cell
/// difference of at least [kMemorySceneChange], or at least
/// [kMemoryChangedCells] cells that differ by at least [kMemoryCellChange].
/// Each cell averages ~900 pixels of a 720p frame, so sensor noise stays
/// well under 1, and the brightness match takes out an exposure or lighting
/// change of the whole view; a new view gives a mean far above
/// [kMemorySceneChange]. A small object moves the mean by well under it but
/// changes a few cells a lot, which the cell test catches.
///
/// Checked by test/data/services/detector/memory_shot_test.dart on a
/// synthetic 720p desk (`SCENE_CHANGE`): an object put down or taken away is
/// a change, with the camera still or moved a few pixels; the camera moved
/// alone, sensor noise, and the whole view darker or brighter are not. The
/// re-alignment keeps a hand-held camera's shake from storing a new frame
/// each time.
const kMemorySceneChange = 6.0;
const kMemoryCellChange = 24.0;
const kMemoryChangedCells = 2;

/// The visual memory is on by default only where the OS reports at least this
/// much RAM in all: a device sold with 6 GB or less gets it off by default,
/// with the reason on the camera settings switch (the user may turn it on). Its
/// image encoder adds memory in steps: the widening reload
/// (`EmbedderService.enableImages`), then the first image. The OS reports
/// less than the RAM fitted, so the line is drawn between what an 8 GB and a
/// 6 GB device report, at 6.5 GiB.
const kMemoryDefaultOnMinRamBytes = 6979321856; // 6.5 GiB

/// The scene-change thumbnail's side, in cells.
const kMemoryGreySide = 32;

/// Each scene-change cell is this many grey pixels a side: the thumbnail is
/// 128×128 (one pixel ≈ 10 × 5.6 px of a 720p frame), fine enough to
/// re-align a hand-held camera's shift before the pixels are averaged into
/// cells.
const kMemoryCellPixels = 4;

/// A hand-held camera moves between two samples: the reference thumbnail
/// is re-aligned by every global shift of up to this many thumbnail pixels
/// in x and y, in half-pixel steps (≈ 20 px across and 11 px down a 720p
/// frame), and the best alignment is compared (`sceneChange`). A larger
/// move is a new frame.
const kMemoryMaxShift = 2;

/// A memory frame's long side, in pixels: what is embedded, kept, shown on
/// the result card and sent to the chat model.
const kMemoryThumbMaxSide = 320;

/// The memory keeps at most this many frames and this many bytes of PNG
/// thumbnails (measured), dropping the oldest first.
const kMemoryMaxEntries = 200;
const kMemoryMaxBytes = 16 << 20;

/// A memory frame must come back from the detector worker within this, or
/// the attempt is dropped.
const kMemoryCaptureTimeout = Duration(seconds: 2);

/// A memory question's best frame is accepted when its cosine is at least
/// [kMemoryMinCosine] and it beats, by at least [kMemoryMinMargin], the best
/// frame that is more than [kMemoryMarginSeparation] away from it in time
/// and shows a different scene (image vectors less alike than
/// [kMemorySameSceneCosine]: frames of one long sighting are not rivals).
/// With no such rival (a session that saw one scene) it must clear the
/// floor plus the margin. A rival within the margin makes the answer
/// ambiguous ("maybe in more than one place", the newest competing frame)
/// when the best clears the floor plus the margin (both sightings then at or
/// above the floor), and a miss when it does not.
///
/// Set by integration_test/visual_memory_test.dart over its fixture scenes
/// (EmbeddingGemma 2 740m at 70 vision tokens): the floor alone rejects
/// every question about a thing never shown and leaves room below the
/// weakest right frame for real camera frames; the margin is about the
/// middle of the gap between the never-shown questions' margins and the
/// smallest margin of a right frame over its best rival (two different
/// signs). [kMemorySameSceneCosine] is the middle of the gap between two
/// frames of one scene (darker, shifted) and frames of different scenes.
const kMemoryMinCosine = 0.62;
const kMemoryMinMargin = 0.025;
const kMemoryMarginSeparation = Duration(seconds: 5);
const kMemorySameSceneCosine = 0.88;

/// Frames of the best one's scene (image cosine ≥ [kMemorySameSceneCosine])
/// whose question cosine is within this of the best are ties, and the newest
/// of them is reported: "seen 12 s ago" comes from the latest sighting, not
/// from whichever copy won by noise. A frame whose grey thumbnail shows a
/// change from the best's (the sampler's scene-change test: keys put down or
/// taken away hardly move the image vector) is never a tie. Above the spread
/// one question scores against a frame, the same frame darker and the same
/// frame shifted (integration_test/visual_memory_test.dart prints it,
/// `MEMORY_TIE`), and below the smallest margin of a right frame over a
/// different scene.
const kMemoryTieCosine = 0.01;

/// Failed frame embeddings in a row that turn the visual memory off (with
/// the last failure as its reason) until it is cleared.
const kMemoryFailuresUntilOff = 3;

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
