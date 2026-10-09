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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;

import '../domain/models/detection.dart' show DetectorBackend;
import '../utils/result.dart';

/// The command line was not understood; [message] says why.
final class SelfTestUsageException implements Exception {
  const SelfTestUsageException(this.message);

  final String message;

  @override
  String toString() => '$message\n\n$kSelfTestUsage';
}

const kSelfTestUsage = '''
Self-test: hardware probe, detector, cats golden, Gemma, detector again, audio.
  --selftest (or SELFTEST=1)
  --gemma=PATH              chat model .litertlm, loaded with the app's Gemma 4 E2B settings
                            (default: the chat model chosen on the Models screen with its saved
                            settings, else GEMMA_MODEL_PATH; the app ships none)
  --detector=PATH           YOLO26n raw-head .tflite (default: the one built into the app)
  --image=PATH              detect on this image instead of the bundled cats (no golden check)
  --gemma-backend=npu|gpu|cpu  default: the chat model's own (gpu for Gemma 4 E2B); never falls
                            back; npu fails with the reason where flutter_edge_ai has no NPU stack
  --detector-backend=gpu|cpu  default gpu; never falls back
  --detector-cpu-retry      after a failed GPU detector load, also try the CPU (information only)
  --allow-software-gpu      a software GPU (llvmpipe) passes, labelled: tests the GPU code path only
  --skip-audio              leave out step 6 (a 440 Hz tone on the default output, recorded from
                            the sink's monitor on Linux; 1 s from the default microphone)
  --out=PATH                report file (default: <app support>/selftest/selftest-<time>.txt)
  --timeout=SECONDS         give up (FAIL, exit 1) when the run takes longer; default 1800
Exit code 0 when every step passed, else 1.''';

/// What `--selftest` was asked to do.
final class const SelfTestOptions({
  final String? gemmaPath,
  final String? detectorPath,
  final String? imagePath,
  final String? outPath,

  /// Null: the chat model's own backend (gpu for Gemma 4 E2B, the saved one
  /// for the user's model).
  final PreferredBackend? gemmaBackend,
  final DetectorBackend detectorBackend = DetectorBackend.gpu,

  /// After a failed detector load on the GPU, try the CPU too, labelled as
  /// information: the self-test still fails.
  final bool detectorCpuRetry = false,

  /// A software GPU (llvmpipe & co.) does not fail a GPU step; it is still
  /// labelled. For checking the GPU code path on a machine without a GPU.
  final bool allowSoftwareGpu = false,

  /// Leave out the audio steps (6a output, 6b microphone): for a headless
  /// machine without a sound server.
  final bool skipAudio = false,

  /// The whole run's limit: past it the watchdog prints what finished and
  /// exits 1 (a native call that never returns must not hang a headless
  /// run). Generous: a first GPU run compiles its programs.
  final Duration timeout = const Duration(minutes: 30),
}) {
  /// Null when no self-test was requested (`--selftest` absent and
  /// `SELFTEST` not `1`); an error for an unknown or malformed option, so a
  /// typo never runs a different test than the one asked for. Arguments not
  /// starting with `--` are ignored: macOS adds its own (`-NSDocument…`).
  static Result<SelfTestOptions>? parse(
    List<String> args,
    Map<String, String> environment,
  ) {
    final requested =
        args.contains('--selftest') || environment['SELFTEST'] == '1';
    if (!requested) return null;
    String? gemma;
    String? detector;
    String? image;
    String? out;
    PreferredBackend? gemmaBackend;
    var detectorBackend = DetectorBackend.gpu;
    var cpuRetry = false;
    var allowSoftware = false;
    var skipAudio = false;
    var timeout = const Duration(minutes: 30);
    for (final arg in args) {
      if (!arg.startsWith('--') || arg == '--selftest') continue;
      final eq = arg.indexOf('=');
      final key = eq < 0 ? arg : arg.substring(0, eq);
      final value = eq < 0 ? null : arg.substring(eq + 1);
      Result<SelfTestOptions> bad(String why) =>
          Result.error(SelfTestUsageException('$arg: $why'));
      if (key
          case '--detector-cpu-retry' ||
              '--allow-software-gpu' ||
              '--skip-audio') {
        if (value != null) return bad('takes no value');
        switch (key) {
          case '--detector-cpu-retry':
            cpuRetry = true;
          case '--allow-software-gpu':
            allowSoftware = true;
          default:
            skipAudio = true;
        }
        continue;
      }
      if (value == null || value.isEmpty) return bad('needs =VALUE');
      switch (key) {
        case '--gemma':
          gemma = value;
        case '--detector':
          detector = value;
        case '--image':
          image = value;
        case '--out':
          out = value;
        case '--gemma-backend':
          switch (value) {
            case 'npu':
              gemmaBackend = PreferredBackend.npu;
            case 'gpu':
              gemmaBackend = PreferredBackend.gpu;
            case 'cpu':
              gemmaBackend = PreferredBackend.cpu;
            default:
              return bad('npu, gpu or cpu');
          }
        case '--timeout':
          final seconds = int.tryParse(value);
          if (seconds == null || seconds <= 0) return bad('whole seconds > 0');
          timeout = Duration(seconds: seconds);
        case '--detector-backend':
          final parsed = DetectorBackend.tryParse(value);
          if (parsed == null) return bad('gpu or cpu');
          detectorBackend = parsed;
        default:
          return bad('unknown option');
      }
    }
    return Result.ok(
      SelfTestOptions(
        gemmaPath: gemma,
        detectorPath: detector,
        imagePath: image,
        outPath: out,
        gemmaBackend: gemmaBackend,
        detectorBackend: detectorBackend,
        detectorCpuRetry: cpuRetry,
        allowSoftwareGpu: allowSoftware,
        skipAudio: skipAudio,
        timeout: timeout,
      ),
    );
  }
}
