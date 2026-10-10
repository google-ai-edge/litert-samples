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

import '../data/services/hardware/native_log_tap.dart';
import '../domain/hardware/diagnostics_report.dart' show evidenceText, seconds;
import '../domain/models/hardware_profile.dart';
import '../utils/result.dart';
import 'evidence_judge.dart';
import 'self_test_ports.dart';
import 'step_recorder.dart';
import 'tapped_load.dart';

/// The prompt step 4 generates from (the coex test's).
const kSelfTestPrompt = 'Tell me a short story about a lighthouse keeper.';

/// Tokens step 4 asks for.
const kSelfTestMaxTokens = 64;

/// Steps 4a and 4b through [SelfTestGemma]: the chat model (Gemma 4 E2B or
/// the user's own `.litertlm`) loaded with exactly its configuration (no
/// fallback) and warmed up, then [kSelfTestMaxTokens] tokens timed.
final class ChatModelSteps {
  ChatModelSteps({
    required this._recorder,
    required this._judge,
    required this._gemma,
    required this._chatModel,
    required this._logTap,
    required this._hardware,
  });

  final StepRecorder _recorder;
  final EvidenceJudge _judge;
  final SelfTestGemma _gemma;
  final SelfTestChatModel _chatModel;
  final NativeLogTap _logTap;

  /// The probed host (step 1); null before it or when the probe failed.
  final HardwareProfile? Function() _hardware;

  bool _softwareGpuAllowed = false;
  bool _loadStarted = false;

  /// The load passed on a software (or unnamed Linux) GPU only because
  /// `--allow-software-gpu` asked for it.
  bool get softwareGpuAllowed => _softwareGpuAllowed;

  /// Step 4a's body has started: a chat model engine may exist from then on
  /// (until the runner closes the chat model).
  bool get loadStarted => _loadStarted;

  /// Steps 4a and 4b; 4b is skipped unless 4a passed.
  Future<void> run() async {
    final config = _chatModel.config;
    var loaded = false;
    await _recorder.run(
      '4a',
      'chat model load + warm-up (${config.llm.backend.name})',
      () async {
        _loadStarted = true;
        final (status, details) = await _load();
        loaded = status == StepStatus.pass;
        return (status, details);
      },
    );
    if (!loaded) {
      _recorder.skip(
        '4b',
        'chat model generate',
        'the chat model did not load as requested',
      );
      return;
    }
    await _recorder.run(
      '4b',
      'chat model generate $kSelfTestMaxTokens tokens',
      _generate,
    );
  }

  Future<StepOutcome> _load() async {
    final config = _chatModel.config;
    final backend = config.llm.backend;
    final path = _chatModel.file.path;
    if (path == null) {
      return (
        StepStatus.fail,
        [_chatModel.file.problem ?? 'no chat model file'],
      );
    }
    final load = await loadWithNativeLog(
      _logTap,
      () => _gemma.load(path, config),
    );
    switch (load.result) {
      case Error(:final error):
        return (StepStatus.fail, load.failure(error));
      case Ok(:final value):
        final evidence = load.evidence(
          requested: backend.name,
          reported: value.backend.name,
          hardware: _hardware(),
        );
        final verdict = _judge.accelerator(
          evidence,
          platform: _hardware()?.platform,
        );
        if (verdict.softwareGpuAllowed) _softwareGpuAllowed = true;
        return (
          verdict.status,
          [
            '${config.name} (${value.modelId}) · ${value.config} · load '
                '${seconds(value.loadTime)} · warm-up '
                '${seconds(value.warmUpTime)}',
            evidenceText(evidence),
            if (config.llm.supportImage)
              'vision encoder on the CPU (flutter_edge_ai default, requested)',
            ...verdict.errors,
            ...load.logLines,
          ],
        );
    }
  }

  Future<StepOutcome> _generate() async {
    switch (await _gemma.generate(kSelfTestPrompt, kSelfTestMaxTokens)) {
      case Error(:final error):
        return (StepStatus.fail, ['generate failed: $error']);
      case Ok(value: final g):
        final excerpt = g.text.replaceAll(RegExp(r'\s+'), ' ').trim();
        final enough = _judge.enoughChunks(g.chunks);
        return (
          enough ? StepStatus.pass : StepStatus.fail,
          [
            '${g.chunks} chunks'
                '${g.engineTokens == null ? '' : ' · ${g.engineTokens} tokens (engine)'} · '
                'first chunk ${seconds(g.firstChunk)} · '
                '${g.decodeRate.toStringAsFixed(1)} tok/s decode (measured)'
                '${g.engineTokensPerSecond == null ? '' : ' · ${g.engineTokensPerSecond!.toStringAsFixed(1)} tok/s (engine)'}',
            if (!enough) 'fewer than $kSelfTestMinChunks chunks',
            '"${excerpt.length > 100 ? '${excerpt.substring(0, 100)}…' : excerpt}"',
          ],
        );
    }
  }
}
