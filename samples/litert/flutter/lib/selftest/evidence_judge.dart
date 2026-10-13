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

import '../domain/models/accelerator_evidence.dart';
import '../domain/models/detection.dart';
import '../domain/models/hardware_profile.dart' show HostPlatform;
import '../utils/pcm.dart';
import 'cats_golden.dart';
import 'step_recorder.dart';

/// Fewer streamed chunks than this is a failed generation.
const kSelfTestMinChunks = 8;

/// The monitor capture's loudest 100 ms must reach this for step 6a to pass.
const kMonitorPassDbfs = -40.0;

/// Below this (or all zeros) nothing reached the sound server; between this
/// and [kMonitorPassDbfs] the tone arrived too quietly.
const kMonitorNothingDbfs = -80.0;

/// A microphone whose loudest 100 ms stays below this is silent: a WARN
/// (the room may be quiet), not a failure.
const kMicSilenceDbfs = -60.0;

/// A load's accelerator evidence, judged.
final class const AcceleratorVerdict({
  required final StepStatus status,

  /// Why the evidence fails the step, or what lets it pass anyway; empty
  /// when it is clean.
  required final List<String> errors,

  /// The step passes only because `--allow-software-gpu` asked for it.
  final bool softwareGpuAllowed = false,
});

/// Step 3's ten detections, judged.
final class const CatsVerdict({
  /// All runs gave the first run's boxes, bit for bit.
  required final bool identical,

  /// The median run time after the first run, in ms.
  required final double medianRunMs,

  /// The first run against [kCatsGolden]; null for an image without one.
  final CatsGoldenCheck? golden,
}) {
  bool get passed => identical && (golden?.passed ?? true);
}

/// What the sink's monitor recorded of step 6a's tone.
enum MonitorVerdict {
  /// Loud enough: it reached the sound server.
  heard,

  /// It arrived, but below [kMonitorPassDbfs].
  tooQuiet,

  /// All zeros or below [kMonitorNothingDbfs].
  nothing,
}

/// What step 6b's microphone delivered.
enum MicVerdict {
  /// Not a sample.
  empty,

  /// Only digital zeros: the input is muted, or the OS hands the app none.
  digitalZeros,

  /// Below [kMicSilenceDbfs]: the room may be quiet.
  silent,

  /// The room is heard.
  heard,
}

/// The self-test's thresholds and comparisons. Pure: no I/O, no clock. The
/// steps phrase the verdicts, except [accelerator]'s errors: those lines are
/// the same for every load, so the judge writes them.
final class const EvidenceJudge({
  /// `--allow-software-gpu`: a software GPU (or, on Linux, an adapter
  /// nothing can name) passes a GPU step, still labelled.
  required final bool allowSoftwareGpu,
}) {
  /// A confirmed mismatch fails the step even when the runtime said yes; so
  /// does a software GPU, or on Linux an adapter nothing can name (a
  /// software GPU cannot be ruled out), unless [allowSoftwareGpu] asked to
  /// test the GPU code path on one (still labelled). [platform] is the
  /// probed host's; null when the probe failed.
  AcceleratorVerdict accelerator(
    AcceleratorEvidence e, {
    required HostPlatform? platform,
  }) {
    // WebGPU on Linux opens llvmpipe as readily as a real GPU: an adapter
    // that neither the log nor the probe names is not a pass.
    final unnamedOnLinux = e.adapterUnknown && platform == HostPlatform.linux;
    final errors = [
      if (e.mismatch) 'requested ${e.requested} but runs on ${e.actual}',
      if (unnamedOnLinux)
        'cannot rule out a software GPU: no "Selected adapter" line in the '
            'native log and the probe found no hardware GPU (install '
            'vulkan-tools for vulkaninfo)'
            '${allowSoftwareGpu ? ' (allowed by --allow-software-gpu)' : ''}',
      if (e.softwareGpu)
        'the GPU is a software rasterizer (${e.adapter ?? 'unknown'}, '
            '${e.adapterSource.label}): this is the CPU, not a GPU'
            '${allowSoftwareGpu ? ' (allowed by --allow-software-gpu: the GPU code path is tested, not GPU speed)' : ''}',
    ];
    if (e.mismatch) {
      return AcceleratorVerdict(status: StepStatus.fail, errors: errors);
    }
    if (!e.softwareGpu && !unnamedOnLinux) {
      return AcceleratorVerdict(status: StepStatus.pass, errors: errors);
    }
    if (!allowSoftwareGpu) {
      return AcceleratorVerdict(status: StepStatus.fail, errors: errors);
    }
    return AcceleratorVerdict(
      status: StepStatus.pass,
      errors: errors,
      softwareGpuAllowed: true,
    );
  }

  /// Step 3: [frames] all identical to the first, and the first against
  /// [kCatsGolden] when [golden]. At least two frames: the median leaves out
  /// the first run (it may compile GPU programs).
  CatsVerdict cats(List<DetectionFrame> frames, {required bool golden}) {
    final runMs = [for (final f in frames.skip(1)) f.runMicros / 1000]..sort();
    return CatsVerdict(
      identical: frames.every((f) => f.sameBoxes(frames.first)),
      medianRunMs: runMs[runMs.length ~/ 2],
      golden: golden ? checkCatsGolden(frames.first) : null,
    );
  }

  /// Step 5: [frame] bit-identical to step 3's [reference].
  bool sameAsReference(DetectionFrame frame, DetectionFrame reference) =>
      frame.sameBoxes(reference);

  /// Step 4b: enough streamed chunks for a real generation.
  bool enoughChunks(int chunks) => chunks >= kSelfTestMinChunks;

  /// Step 6a: the monitor capture's level.
  MonitorVerdict monitor(PcmStats stats) {
    if (stats.peakFrameDbfs >= kMonitorPassDbfs) return MonitorVerdict.heard;
    return stats.allZero || stats.peakFrameDbfs < kMonitorNothingDbfs
        ? MonitorVerdict.nothing
        : MonitorVerdict.tooQuiet;
  }

  /// Step 6b: the microphone capture's level.
  MicVerdict mic(PcmStats stats) {
    if (stats.samples == 0) return MicVerdict.empty;
    if (stats.allZero) return MicVerdict.digitalZeros;
    if (stats.peakFrameDbfs < kMicSilenceDbfs) return MicVerdict.silent;
    return MicVerdict.heard;
  }
}
