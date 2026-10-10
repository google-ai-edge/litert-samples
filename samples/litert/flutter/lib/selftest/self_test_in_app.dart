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

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import '../config/build_info.dart';
import '../config/env.dart';
import '../data/repositories/audio_repository.dart';
import '../data/services/hardware/hardware_info_service.dart';
import '../data/services/hardware/memory_probe.dart';
import '../data/services/hardware/native_log_tap.dart';
import '../domain/models/hardware_profile.dart' show HostPlatform;
import '../domain/models/model_id.dart';
import '../domain/models/model_state.dart';
import '../domain/models/self_test.dart';
import '../domain/ports/chat_model_planner.dart';
import '../domain/ports/self_test_launcher.dart';
import '../domain/use_cases/chat_model_switcher.dart';
import '../utils/result.dart';
import 'self_test_adapters.dart';
import 'self_test_options.dart';
import 'self_test_report.dart';
import 'self_test_runner.dart';

/// How long a run past its time limit gets to skip its remaining steps and
/// close its models before the app gives up on it.
const kSelfTestStopGrace = Duration(seconds: 10);

/// The line the report and the progress get when a timed-out run did not
/// end within [kSelfTestStopGrace] and may still hold its chat model engine.
const kSelfTestStillRunning =
    'the self-test is still running; restart the app before using the chat '
    'model';

/// [kSelfTestStillRunning] for a run that holds no chat model engine: it
/// hung before its chat model load, or after closing it (an audio close
/// that outlives the grace). The app's chat model is loaded again.
const kSelfTestStillRunningNoChatModel =
    'the self-test is still running, but it no longer holds a chat model '
    'engine (the app\'s is loaded again); restart the app before running the '
    'self-test again';

/// Waits up to [limit] for [run] (a [SelfTestRunner.run]). Past it: the
/// report so far ([partial]), [stop] asks the runner to skip its remaining
/// steps, and up to [grace] is given for [run] to end (the runner closes its
/// models then). Still not ended: `stillRunning`, and the caller must not
/// load another engine while the runner may hold one
/// ([SelfTestRunner.chatEngineMayBeLoaded]).
@visibleForTesting
Future<({R report, bool stillRunning})> runSelfTestWithin<R>(
  Future<R> run, {
  required Duration limit,
  required Duration grace,
  required R Function() partial,
  required void Function() stop,
}) async {
  var timedOut = false;
  final report = await run.timeout(
    limit,
    onTimeout: () {
      timedOut = true;
      return partial();
    },
  );
  if (!timedOut) return (report: report, stillRunning: false);
  stop();
  var ended = true;
  await run
      .then<void>(
        (_) {},
        onError: (Object e, StackTrace st) =>
            debugPrint('[SelfTest] the timed-out run failed: $e\n$st'),
      )
      .timeout(grace, onTimeout: () => ended = false);
  if (!ended) {
    debugPrint(
      '[SelfTest] the timed-out run did not end within ${grace.inSeconds} s',
    );
  }
  return (report: report, stillRunning: !ended);
}

/// The steps of one in-app run: [progress] gets each line; [stopped]
/// completes when the app closes ([InAppSelfTest.close]), and the runner is
/// then asked to stop.
typedef SelfTestSteps = Future<SelfTestOutcome> Function(
  void Function(String line) progress,
  Future<void> stopped,
);

/// The `--selftest` runner inside the running app (Android has no command
/// line): the same steps through the same adapters, on the chat model chosen
/// on the Models screen with its own backend. flutter_edge_ai holds one
/// model per process, so the app's chat is released and its chat model
/// unloaded for the run and loaded again after it — all inside one
/// [ChatModelSwitcher.exclusive], so no reload, model setup or other run can
/// touch the engine meanwhile. The audio steps use the app's own audio
/// repository (soloud is one engine per process); Linux measures the output
/// from the sink's monitor, elsewhere it is played and the device checked.
final class InAppSelfTest implements SelfTestLauncher {
  InAppSelfTest({
    required this._switcher,
    required this._chatModels,
    required this._models,
    required this._logTap,
    required this._audio,
    this._options = const SelfTestOptions(),
    @visibleForTesting this._steps,
  });

  final ChatModelSwitcher _switcher;
  final ChatModelPlanner _chatModels;
  final ValueListenable<Map<ModelId, ModelState>> _models;
  final NativeLogTap _logTap;
  final AudioRepository _audio;
  final SelfTestOptions _options;

  /// Replaces the real steps (tests of the unload/reload around them). They
  /// get the progress sink and [close]'s stop request.
  final SelfTestSteps? _steps;
  bool _running = false;

  /// Completed when the run in flight has ended; null when none runs.
  Completer<void>? _ended;

  /// Completed by [close]: the run in flight is asked to stop.
  final Completer<void> _closing = Completer<void>();

  /// A timed-out run never ended: its engine may still be loaded, so no
  /// other run may load one until the app restarts.
  bool _stuck = false;

  @override
  bool get stuck => _stuck;

  @override
  Future<Result<SelfTestOutcome>> run({
    required void Function(String line) progress,
  }) async {
    if (_running) {
      return Result.error(
        asException(StateError('A self-test is already running')),
      );
    }
    if (_stuck) {
      return Result.error(asException(StateError(kSelfTestStillRunning)));
    }
    if (_closing.isCompleted) return Result.error(_closed());
    _running = true;
    final ended = _ended = Completer<void>();
    try {
      return await _switcher.exclusive((ops) async {
        // The app began closing while this run waited for the chat model.
        if (_closing.isCompleted) return Result.error(_closed());
        final before = _models.value[ModelId.chat] ?? const ModelPending();
        var stillRunning = false;
        var holdsChatEngine = false;
        var refused = false;
        try {
          progress('releasing the chat and unloading the chat model …');
          if (await ops.unload() case Error(:final error)) {
            // The chat's stopped reply still generates: nothing was
            // released or unloaded, so nothing is loaded again either.
            refused = true;
            return Result.error(error);
          }
          final outcome = await (_steps ?? _runSteps)(
            progress,
            _closing.future,
          );
          stillRunning = outcome.stillRunning;
          holdsChatEngine = stillRunning && outcome.chatEngineMayBeLoaded;
          return Result.ok(outcome);
        } catch (e, st) {
          debugPrint('[SelfTest] in-app run failed: $e\n$st');
          return Result.error(asException(e));
        } finally {
          // The app's chat model as it was: loaded again after a run that
          // found it loaded or failed (a failure is then shown again with
          // its reason) — unless the run's own engine may still be loaded:
          // then nothing loads one (no setup run, no reload) until a
          // restart. A refused unload left it loaded: nothing to load. The
          // app closing ([close]) loads nothing either.
          if (stillRunning) {
            _stuck = true;
            progress(
              holdsChatEngine
                  ? kSelfTestStillRunning
                  : kSelfTestStillRunningNoChatModel,
            );
          }
          if (holdsChatEngine) {
            ops.refuseLoads(kSelfTestStillRunning);
          } else if (!refused &&
              before is! ModelPending &&
              !_closing.isCompleted) {
            progress('loading the app\'s chat model again …');
            await ops.reload();
          }
        }
      });
    } finally {
      _running = false;
      _ended = null;
      ended.complete();
    }
  }

  /// The app is closing (`AppDependencies.dispose`): a run in flight is
  /// asked to stop — the step in flight ends on its own (a native call
  /// cannot be interrupted), the later ones are skipped and the runner
  /// closes its models — and the app's chat model is not loaded again after
  /// it. Waits up to [grace] for the run to end; no run starts afterwards.
  Future<void> close({Duration grace = kSelfTestStopGrace}) async {
    if (!_closing.isCompleted) _closing.complete();
    final ended = _ended;
    if (ended == null) return;
    debugPrint('[SelfTest] the app closes: stopping the run in flight');
    var inTime = true;
    await ended.future.timeout(grace, onTimeout: () => inTime = false);
    if (!inTime) {
      debugPrint(
        '[SelfTest] the run did not end within ${grace.inSeconds} s of the '
        'stop; the app closes anyway',
      );
    }
  }

  static Exception _closed() =>
      asException(StateError('The app is closing: no self-test runs'));

  Future<SelfTestOutcome> _runSteps(
    void Function(String) progress,
    Future<void> stopped,
  ) async {
    final platform = HostPlatform.fromOperatingSystem(Platform.operatingSystem);
    final chatModel = await resolveSelfTestChatModel(
      argument: null,
      define: kGemmaModelPath,
      plan: _chatModels.plan,
    );
    final detectorFile = await resolveSelfTestDetector(argument: null);
    final runner = SelfTestRunner(
      options: _options,
      build: currentBuildInfo(),
      hardware: hardwareInfoServiceForPlatform(),
      detector: AppSelfTestDetector(),
      gemma: AppSelfTestGemma(),
      memory: memoryProbeFor(platform),
      logTap: _logTap,
      chatModel: chatModel,
      detectorFile: detectorFile,
      loadImage: () => loadSelfTestImage(null),
      imageLabel: kSelfTestCatsLabel,
      audio: _options.skipAudio
          ? null
          : AppSelfTestAudio(platform, shared: _audio),
      progress: progress,
    );
    // The app closing stops the runner; the holder (not the runner) is what
    // the listener keeps once the run has ended.
    SelfTestRunner? live = runner;
    unawaited(stopped.then((_) => live?.stop()));
    // A native call that never returns must not hold the button forever:
    // past the limit, the report so far with a failed timeout step. The
    // runner is stopped and given a grace period to close its own engine.
    final ({SelfTestReport report, bool stillRunning}) within;
    try {
      within = await runSelfTestWithin(
        runner.run(),
        limit: _options.timeout,
        grace: kSelfTestStopGrace,
        partial: () => runner.report(timedOut: _options.timeout),
        stop: runner.stop,
      );
    } finally {
      live = null;
    }
    final (:report, :stillRunning) = within;
    final holdsChatEngine = runner.chatEngineMayBeLoaded;
    final note = switch ((stillRunning, holdsChatEngine)) {
      (false, _) => null,
      (true, true) => kSelfTestStillRunning,
      (true, false) => kSelfTestStillRunningNoChatModel,
    };
    final path = await _write(report, note: note);
    final text = _withNote(formatSelfTest(report, reportPath: path), note);
    debugPrint(text);
    return SelfTestOutcome(
      text: text,
      passed: report.passed,
      reportPath: path,
      stillRunning: stillRunning,
      chatEngineMayBeLoaded: holdsChatEngine,
    );
  }

  /// [block] with [note] on the line before it.
  static String _withNote(String block, String? note) =>
      note == null ? block : '$note\n$block';

  /// `<app support>/selftest/selftest-<time>.txt` (on Android
  /// `adb shell run-as com.google.ai.edge.examples.litert_edge_demos cat …`);
  /// null when it cannot be written.
  Future<String?> _write(SelfTestReport report, {String? note}) async {
    try {
      final stamp = DateTime.now()
          .toUtc()
          .toIso8601String()
          .split('.')
          .first
          .replaceAll(':', '-');
      final file = File(
        '${(await getApplicationSupportDirectory()).path}/selftest/'
        'selftest-$stamp.txt',
      );
      await file.parent.create(recursive: true);
      await file.writeAsString(
        _withNote(formatSelfTest(report, reportPath: file.path), note),
      );
      return file.path;
    } on FileSystemException catch (e) {
      debugPrint('[SelfTest] could not write the report: $e');
      return null;
    }
  }
}
