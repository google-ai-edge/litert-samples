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

// On the real app (macOS): Demo 3's fast path (moonshine STT,
// no LLM) and its detailed path (the frozen frame to Gemma), a barge-in, and
// Demo 1 after switching the recognizer back to Whisper. YOLO26n on the GPU
// over fixture images, moonshine / Whisper / Inflect on the CPU, Gemma 4 E2B
// on the GPU.
//
//   flutter test integration_test/camera_assistant_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm> \
//     --dart-define=FRAME_SOURCE=fixture \
//     --dart-define=FIXTURE_DIR=<path/to/test_assets>/cats.jpg \
//     --dart-define=ROUTER_AUDIO_DIR=<path/to/router_audio>   # optional, step 8
//
// FIXTURE_DIR is the cats image (COCO val2017 39769) as a file. The sandboxed
// macOS debug build reads files only from its container and ~/Downloads:
// there, pass a copy under ~/Downloads (e.g. of the whole test_assets/).
//
// 1. setup → home → Demo 1 (Whisper switched in) → home → Demo 3 (moonshine
//    switched in, in parallel with the chat); the cats fixture runs live.
// 2. Fast: the real mic button, held while the fixture mic plays
//    test_assets/q_cats.wav, then released. The capture starts 1.5 s after
//    the press (a cold audio warm-up, played): "Opening the mic…" until
//    then, and the clip is timed from Listening, as a user would. "I count
//    two cats.", no LLM turn, first audio ≤ 1.0 s from release.
//    `MIC_OPEN press_to_listening=… capture_to_listening=… …`
//    `CAMQ route=fast rule=count stt_model=… stt=… first_audio=… …`
// 3. Detailed on the GATE 42 sign (1280×720: the first image after the
//    32×32 warm-up is 16:9): "What does the sign say?" contains "42"; the
//    view froze on the frame that was encoded; PNG long side ≤ 1024; the
//    detector paused during generation and back to ≥ 12 fps within 1 s.
//    `CAMQ route=detailed png=…ms reset=…ms ttft=… first_audio=… paused=…ms
//    recover=…ms …`
// 4. The same through the fixture mirrored like camera_desktop: still "42",
//    and the PNG has the arrow back on the right (un-mirrored).
// 5. Cats: "Describe the scene." mentions 2 of {cat, remote, sofa/couch}.
// 6. Barge-in during a spoken detailed answer: silenced ≤ 150 ms, unfrozen at
//    once, the detector recovers.
// 7. Back to Demo 1: Whisper active again, "What is the capital of France?"
//    → Paris. `DEMO1 stt_model=… switch=… stt=… first_audio=…`
// 8. With ROUTER_AUDIO_DIR, a folder holding index.json (a copy of
//    test_assets/router_golden.json) and `<i>.pcm`, item i spoken as 16 kHz
//    mono PCM16 without a header: every router golden question transcribed
//    by the app's moonshine and routed: `ROUTER_REAL …`.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/question_router.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/ui/core/level_meter.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:litert_edge_demos/utils/pcm.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/demo3_settings.dart';
import 'support/fixtures.dart';
import 'support/pump.dart';
import 'support/wav.dart';

const kRouterAudioDir = String.fromEnvironment('ROUTER_AUDIO_DIR');

final _screenKey = GlobalKey();

Future<String> saveScreenshot(String name) async {
  final boundary =
      _screenKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
  if (boundary == null) fail('No RepaintBoundary to capture');
  final image = await boundary.toImage(pixelRatio: 2);
  final png = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  if (png == null) fail('PNG encoding failed');
  final file = File('${Directory.systemTemp.path}/$name.png');
  await file.writeAsBytes(png.buffer.asUint8List());
  return file.path;
}

String ms(Duration? d) => d == null ? '–' : '${d.inMilliseconds}';

Future<Uint8List> asset(String path) async =>
    (await rootBundle.load(path)).buffer.asUint8List();

/// What the detector's pause names (`GpuArbiter`): the loaded chat model.
String _pauseReason(AppDependencies deps) =>
    switch (deps.models.states.value[ModelId.chat]) {
      ModelReady(:final info) => info.chat?.name ?? kDefineChatModel.name,
      _ => 'the chat model',
    };

/// What happened to the detector and the chat while a turn ran, on one
/// clock: the pause, the resume, every detected frame.
final class Timeline {
  Timeline(this._deps) {
    _deps.live.state.addListener(_onState);
    _deps.live.frames.addListener(_onFrame);
    _deps.conversation.isGenerating.addListener(_onGenerating);
  }

  final AppDependencies _deps;
  final Stopwatch _clock = Stopwatch()..start();
  int? pausedAt;
  int? resumedAt;
  int? generationStart;
  int? generationEnd;
  int framesWhilePaused = 0;
  final List<int> framesAfterResume = [];

  int get _now => _clock.elapsedMilliseconds;

  void _onState() {
    switch (_deps.live.state.value) {
      case LivePaused(:final reason) when reason == _pauseReason(_deps):
        pausedAt ??= _now;
      case LiveRunning() when pausedAt != null:
        resumedAt ??= _now;
      default:
        break;
    }
  }

  void _onFrame() {
    if (_deps.live.frames.value == null) return;
    if (pausedAt != null && resumedAt == null) framesWhilePaused++;
    if (resumedAt != null) framesAfterResume.add(_now);
  }

  void _onGenerating() {
    if (_deps.conversation.isGenerating.value) {
      generationStart ??= _now;
    } else if (generationStart != null) {
      generationEnd ??= _now;
    }
  }

  /// Paused for this long (null: never paused or not resumed yet).
  int? get paused =>
      pausedAt != null && resumedAt != null ? resumedAt! - pausedAt! : null;

  /// From the resume to the 12th frame after it: ≤ 1000 means ≥ 12 fps
  /// within a second.
  int? get recover => framesAfterResume.length >= 12
      ? framesAfterResume[11] - resumedAt!
      : null;

  /// Frames detected in the first second after the resume.
  int get framesFirstSecond => resumedAt == null
      ? 0
      : framesAfterResume.where((t) => t - resumedAt! <= 1000).length;

  void dispose() {
    _deps.live.state.removeListener(_onState);
    _deps.live.frames.removeListener(_onFrame);
    _deps.conversation.isGenerating.removeListener(_onGenerating);
  }
}

void main() {
  initIntegrationTest();

  testWidgets('camera assistant: fast path (moonshine, no LLM), detailed path '
      '(frozen frame → Gemma), barge-in, Demo 1 after the switch back', (
    tester,
  ) async {
    if (kGemmaModelPath.isEmpty) fail('Pass GEMMA_MODEL_PATH');
    expect(kFrameSource, 'fixture', reason: 'FRAME_SOURCE=fixture');
    expect(
      kFixtureDir.endsWith('cats.jpg'),
      isTrue,
      reason: 'FIXTURE_DIR must be the cats image',
    );
    final qCats = pcm16FromWav(await asset('test_assets/q_cats.wav'));
    final qSign = pcm16FromWav(await asset('test_assets/q_sign.wav'));
    final qDescribe = pcm16FromWav(await asset('test_assets/q_describe.wav'));
    final qDetail = pcm16FromWav(
      await asset('test_assets/q_describe_detail.wav'),
    );
    final france = await asset('test_assets/france_16k.pcm');
    // The sign, where the sandboxed app may read it.
    final gatePath = '${Directory.systemTemp.path}/gate42.png';
    final gatePng = await asset('test_assets/gate42.png');
    await File(gatePath).writeAsBytes(gatePng);

    final mic = FixtureMicService(qCats);
    await resetDemo3Settings();
    final deps = await AppDependencies.create(mic: mic);
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // 1. Setup, home.
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 8),
      reason: 'model setup',
    );
    for (final id in [
      ModelId.chat,
      ModelId.whisperBase,
      ModelId.inflectNano,
      ModelId.yolo26n,
      ModelId.moonshineTiny,
    ]) {
      expect(deps.models.states.value[id], isA<ModelReady>(), reason: '$id');
    }
    String sttRow(ModelId id) => switch (deps.models.states.value[id]) {
      ModelReady(:final info) =>
        '${info.modelId} load=${info.loadTime.inMilliseconds}ms '
            'warm=${info.warmUpTime.inMilliseconds}ms',
      final other => '$other',
    };
    debugPrint(
      'SETUP stt whisper ${sttRow(ModelId.whisperBase)} | moonshine '
      '${sttRow(ModelId.moonshineTiny)} | active '
      '${deps.activeStt.value?.modelId}',
    );
    await tester.pump(const Duration(milliseconds: 400));

    Future<void> openDemo(Demo demo, Type screen) async {
      await tester.tap(find.byKey(HomeKeys.tile(demo)));
      await pumpUntil(
        tester,
        () => find.byType(screen).evaluate().isNotEmpty,
        timeout: const Duration(seconds: 5),
        reason: 'the ${demo.name} screen to open',
      );
    }

    Future<void> backHome() async {
      await tester.pageBack();
      await pumpUntil(
        tester,
        () =>
            find.byType(HomeScreen).evaluate().isNotEmpty &&
            find.byType(LiveCameraScreen).evaluate().isEmpty &&
            find.byType(VoiceChatScreen).evaluate().isEmpty,
        timeout: const Duration(seconds: 5),
        reason: 'back on home',
      );
      await tester.pump(const Duration(milliseconds: 400));
    }

    // Demo 1 first, so entering Demo 3 switches Whisper → moonshine.
    await openDemo(Demo.voiceChat, VoiceChatScreen);
    await pumpUntil(
      tester,
      () => deps.activeStt.value?.id == ModelId.whisperBase,
      timeout: const Duration(seconds: 10),
      reason: 'Whisper switched in for Demo 1',
    );
    final toWhisper = deps.activeStt.value!.switchTime;
    await backHome();

    await openDemo(Demo.liveCamera, LiveCameraScreen);
    final vm = Provider.of<LiveCameraViewModel>(
      tester.element(find.byType(LiveCameraScreen)),
      listen: false,
    );
    await pumpUntil(
      tester,
      () => deps.activeStt.value?.id == ModelId.moonshineTiny && vm.chatReady,
      timeout: const Duration(seconds: 15),
      reason: 'moonshine switched in and the camera chat open',
      describe: () =>
          'stt=${deps.activeStt.value?.modelId} '
          'sttError=${vm.sttError} chatError=${vm.chatError}',
    );
    final toMoonshine = deps.activeStt.value!.switchTime;
    debugPrint(
      'STT_SWITCH moonshine→whisper=${ms(toWhisper)}ms (Demo 1 entry, no '
      'warm-up) whisper→moonshine=${ms(toMoonshine)}ms (Demo 3 entry)',
    );

    Future<void> waitLive(String what) async {
      var detections = 0;
      void onFrame() {
        if (deps.live.frames.value != null) detections++;
      }

      deps.live.frames.addListener(onFrame);
      try {
        await pumpUntil(
          tester,
          () => deps.live.state.value is LiveRunning && detections >= 8,
          timeout: const Duration(seconds: 20),
          reason: '$what to run live',
          describe: () =>
              'state=${deps.live.state.value} start=${vm.startError}',
        );
      } finally {
        deps.live.frames.removeListener(onFrame);
      }
    }

    await waitLive('the cats fixture');
    final DetectionFrame live = deps.live.frames.value!;
    debugPrint(
      'LIVE frame=${live.frameId} boxes='
      '${[for (var i = 0; i < live.count; i++) '${cocoName(live.classId(i))} ${live.score(i).toStringAsFixed(2)}'].join(', ')}',
    );

    /// Holds the real mic button while the fixture mic plays [pcm], then
    /// releases it; returns the release clock. The hold is timed from what
    /// the user sees: Listening, shown only once the capture runs (the
    /// start waits for the audio warm-up begun at the demo's entry; a cold
    /// output start on macOS took up to 2.6 s). [slowStart]: the start is
    /// delayed, so "Opening the mic…" must show first.
    Future<Stopwatch> askAloud(Uint8List pcm, {bool slowStart = false}) async {
      mic.pcm = pcm;
      final startsBefore = mic.starts;
      final pressed = Stopwatch()..start();
      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(LiveCameraKeys.mic)),
      );
      var openingShown = false;
      await pumpUntil(
        tester,
        () {
          if (vm.isOpeningMic &&
              find.text(kOpeningMicLabel).evaluate().isNotEmpty) {
            openingShown = true;
          }
          return vm.isListening;
        },
        timeout: const Duration(seconds: 10),
        reason: 'the press to show Listening',
        describe: () => 'phase=${vm.phase} mic starts=${mic.starts}',
      );
      final listeningAt = DateTime.now();
      final pressToListening = pressed.elapsed;
      expect(
        mic.starts,
        greaterThan(startsBefore),
        reason: 'Listening is shown only once the capture runs',
      );
      debugPrint(
        'MIC_OPEN press_to_listening=${pressToListening.inMilliseconds}ms '
        'capture_to_listening='
        '${listeningAt.difference(mic.startedAt!).inMilliseconds}ms '
        'opening_shown=$openingShown',
      );
      if (slowStart) {
        expect(
          openingShown,
          isTrue,
          reason: 'a slow start shows "Opening the mic…" until it runs',
        );
      }
      // The clip plus ~0.5 s, as a person releases after the last word.
      await pumpFor(
        tester,
        pcm16Duration(pcm.length, 16000) + const Duration(milliseconds: 550),
      );
      final release = Stopwatch()..start();
      await gesture.up();
      return release;
    }

    Future<void> untilAnswered(String what) => pumpUntil(
      tester,
      () => vm.phase == TurnPhase.idle && vm.exchange.answer != null,
      timeout: const Duration(seconds: 60),
      reason: what,
      describe: () =>
          'phase=${vm.phase} q=${vm.exchange.question} '
          'a=${vm.exchange.answer} notice=${vm.exchange.notice}',
    );

    // 2. Fast path with moonshine.
    final llmBefore = deps.diagnostics.latest.llmTurns;
    var generated = false;
    void onGenerating() {
      if (deps.conversation.isGenerating.value) generated = true;
    }

    deps.conversation.isGenerating.addListener(onGenerating);
    // The first press may land while the audio warm-up begun at the demo's
    // entry still runs (a cold output start on macOS took 2.6 s): the
    // capture then starts well after the press. Played here every run: the
    // user, who starts talking at Listening, must still be heard whole.
    mic.startDelay = const Duration(milliseconds: 1500);
    final fastRelease = await askAloud(qCats, slowStart: true);
    await untilAnswered('the fast answer');
    // The test's wall clock around pumps: a hidden window stretches it
    // (pumps wait for frames). The CAMQ lines print the app's own
    // release-to-drained figure next to it as turn_total.
    final releaseToIdle = fastRelease.elapsed;
    deps.conversation.isGenerating.removeListener(onGenerating);
    {
      final exchange = vm.exchange;
      final voice = deps.diagnostics.latest.lastVoiceTurn!;
      final camera = deps.diagnostics.latest.lastCameraTurn!;
      expect(exchange.question?.toLowerCase(), contains('cats'));
      expect(exchange.answer, 'I count two cats.');
      expect(exchange.route, contains('detector, no LLM'));
      expect(camera.route, isA<FastRoute>());
      expect(camera.route.rule, 'count');
      expect(generated, isFalse, reason: 'the chat never generated');
      final llmTurns = deps.diagnostics.latest.llmTurns - llmBefore;
      expect(llmTurns, 0);
      final firstAudio = voice.firstAudio!;
      debugPrint(
        'CAMQ route=fast rule=${camera.route.rule} '
        'stt_model=${deps.activeStt.value?.modelId} '
        'stt=${ms(voice.stt)}ms first_audio=${ms(firstAudio)}ms '
        'capture_close=${ms(voice.captureClose)}ms '
        'snap=${ms(camera.snapshotLatency)}ms '
        'route_us=${camera.routeTime.inMicroseconds} '
        'tts=${voice.ttsClauses.map((d) => d.inMilliseconds).toList()}ms '
        'release_to_idle=${releaseToIdle.inMilliseconds}ms '
        'turn_total=${ms(voice.total)}ms '
        'llm_turns=$llmTurns basis="${camera.basis}" '
        'transcript="${exchange.question}" answer="${exchange.answer}"',
      );
      expect(
        firstAudio,
        lessThanOrEqualTo(const Duration(milliseconds: 1000)),
        reason: 'fast question: first audio ≤ 1.0 s from release',
      );
    }
    await pumpFor(tester, const Duration(milliseconds: 400));

    /// One detailed question: freezes, asks Gemma, speaks, unfreezes.
    /// Returns the answer; prints the CAMQ line.
    Future<String> detailed(
      String label,
      Uint8List question, {
      String? screenshot,
    }) async {
      final timeline = Timeline(deps);
      final resetBefore = deps.diagnostics.latest.lastCameraReset;
      final release = await askAloud(question);
      int? frozenId;
      int? frozenSize;
      await pumpUntil(
        tester,
        () {
          if (vm.frozen.value case final frozen?) {
            frozenId ??= frozen.frameId;
            frozenSize ??= frozen.image.width;
          }
          return frozenId != null && deps.conversation.isGenerating.value;
        },
        timeout: const Duration(seconds: 20),
        reason: '$label: the view to freeze while Gemma generates',
        describe: () =>
            'phase=${vm.phase} q=${vm.exchange.question} '
            'notice=${vm.exchange.notice} route=${vm.exchange.route}',
      );
      expect(find.byKey(LiveCameraKeys.frozenLabel), findsOneWidget);
      if (screenshot != null) {
        await pumpFor(tester, const Duration(milliseconds: 300));
        debugPrint('SCREENSHOT ${await saveScreenshot(screenshot)}');
      }
      await untilAnswered('$label: the detailed answer');
      final releaseToIdle = release.elapsed;
      expect(vm.frozen.value, isNull, reason: 'spoken: live again');
      // The detector's resume and a second of frames after it; the reset.
      await pumpUntil(
        tester,
        () =>
            timeline.resumedAt != null &&
            timeline.framesFirstSecond >= 12 &&
            !identical(deps.diagnostics.latest.lastCameraReset, resetBefore),
        timeout: const Duration(seconds: 10),
        reason: '$label: the detector back and the chat reset',
        describe: () =>
            'state=${deps.live.state.value} paused=${timeline.pausedAt} '
            'resumed=${timeline.resumedAt} '
            'frames=${timeline.framesAfterResume.length}',
      );
      await pumpFor(tester, const Duration(milliseconds: 300));
      timeline.dispose();

      final sent = vm.lastSentImage!;
      final voice = deps.diagnostics.latest.lastVoiceTurn!;
      final camera = deps.diagnostics.latest.lastCameraTurn!;
      final generation = deps.diagnostics.latest.lastGeneration!;
      expect(camera.route, isA<DetailedRoute>());
      expect(sent.frameId, frozenId, reason: 'encoded frame = frozen frame');
      expect(
        sent.width > sent.height ? sent.width : sent.height,
        lessThanOrEqualTo(kLlmImageMaxSide),
      );
      expect(timeline.pausedAt, isNotNull, reason: 'paused during generation');
      // The frame already in flight when the pause began still lands;
      // nothing new is sent while paused.
      expect(
        timeline.framesWhilePaused,
        lessThanOrEqualTo(1),
        reason: 'no new frames while paused',
      );
      expect(
        timeline.recover,
        lessThanOrEqualTo(1000),
        reason: '≥ 12 fps within 1 s of the resume',
      );
      final answer = vm.exchange.answer!;
      debugPrint(
        'CAMQ route=detailed label=$label rule=${camera.route.rule} '
        'png=${sent.encodeTime.inMilliseconds}ms '
        'png_size=${sent.width}x${sent.height} png_bytes=${sent.png.length} '
        'unmirrored=${sent.unmirrored} frame=${sent.frameId} '
        'frozen=$frozenId frozen_w=$frozenSize '
        'reset=${ms(deps.diagnostics.latest.lastCameraReset)}ms '
        'ttft=${ms(generation.timeToFirstToken)}ms '
        'image_tokens=${generation.imageTokens} '
        'stt=${ms(voice.stt)}ms first_audio=${ms(voice.firstAudio)}ms '
        'snap=${ms(camera.snapshotLatency)}ms '
        'paused=${timeline.paused}ms recover=${timeline.recover}ms '
        'frames_while_paused=${timeline.framesWhilePaused} '
        'frames_1s=${timeline.framesFirstSecond} '
        'gen=${timeline.generationStart}–${timeline.generationEnd}ms '
        'release_to_idle=${releaseToIdle.inMilliseconds}ms '
        'turn_total=${ms(voice.total)}ms '
        'transcript="${vm.exchange.question}" answer="$answer"',
      );
      return answer;
    }

    // 3. The sign, 16:9.
    final started = await deps.live.start(
      FixtureSourceSpec([gatePath]),
      owner: vm,
    );
    expect(started, isA<Ok<FrameSourceInfo>>());
    await waitLive('the GATE 42 sign');
    final sign = await detailed(
      'sign',
      qSign,
      screenshot: 'camera_assistant_frozen',
    );
    expect(sign, contains('42'));
    expect(vm.lastSentImage!.width, 1024);
    expect(vm.lastSentImage!.height, 576);
    expect(vm.lastSentImage!.unmirrored, isFalse);

    // 4. The sign through a source mirrored like camera_desktop.
    await deps.live.start(
      FixtureSourceSpec([gatePath], mirrored: true),
      owner: vm,
    );
    await waitLive('the mirrored sign');
    expect(deps.live.sourceInfo?.mirrored, isTrue);
    final mirrored = await detailed('sign-mirrored', qSign);
    expect(mirrored, contains('42'));
    final sent = vm.lastSentImage!;
    expect(sent.unmirrored, isTrue);
    // The arrow is on the right of the real sign: back there in the PNG.
    final decoded = await tester.runAsync(() async {
      final codec = await ui.instantiateImageCodec(sent.png);
      final image = (await codec.getNextFrame()).image;
      final rgba = (await image.toByteData())!;
      final w = image.width;
      (int, int, int) at(int x, int y) {
        final o = (y * w + x) * 4;
        return (rgba.getUint8(o), rgba.getUint8(o + 1), rgba.getUint8(o + 2));
      }

      final result = (right: at(840, 220), left: at(184, 220));
      image.dispose();
      codec.dispose();
      return result;
    });
    debugPrint('MIRROR png right=${decoded!.right} left=${decoded.left}');
    final (r, g, b) = decoded.right;
    expect(r > 200 && g > 150 && b < 90, isTrue, reason: 'yellow arrow right');

    // 5. Cats: "Describe the scene."
    await deps.live.start(const FixtureSourceSpec([kFixtureDir]), owner: vm);
    await waitLive('the cats fixture');
    final scene = (await detailed('cats', qDescribe)).toLowerCase();
    final mentioned = [
      if (scene.contains('cat')) 'cat',
      if (scene.contains('remote')) 'remote',
      if (scene.contains('sofa') || scene.contains('couch')) 'sofa/couch',
    ];
    debugPrint('SCENE mentions=$mentioned');
    expect(mentioned.length, greaterThanOrEqualTo(2), reason: scene);

    // 6. Barge-in during a spoken detailed answer.
    {
      final timeline = Timeline(deps);
      await askAloud(qDetail);
      await pumpUntil(
        tester,
        () => vm.phase == TurnPhase.speaking && vm.frozen.value != null,
        timeout: const Duration(seconds: 30),
        reason: 'the detailed answer to be spoken over the frozen frame',
        describe: () => 'phase=${vm.phase} frozen=${vm.frozen.value?.frameId}',
      );
      // At once: Gemma is usually still generating the later sentences, so
      // the stop reaches native generation too.
      final generatingAtBargeIn = deps.conversation.isGenerating.value;
      mic.pcm = Uint8List(0);
      final press = await tester.startGesture(
        tester.getCenter(find.byKey(LiveCameraKeys.mic)),
      );
      expect(vm.frozen.value, isNull, reason: 'unfrozen at the press');
      await pumpFor(tester, const Duration(milliseconds: 150));
      await press.up(); // too short: "Didn't catch that"
      await pumpUntil(
        tester,
        () => vm.phase == TurnPhase.idle || vm.phase == TurnPhase.error,
        timeout: const Duration(seconds: 15),
        reason: 'the barge-in to settle',
      );
      await pumpUntil(
        tester,
        () =>
            deps.diagnostics.latest.lastBargeIn?.interruptDone != null &&
            deps.live.state.value is LiveRunning,
        timeout: const Duration(seconds: 15),
        reason: 'the interrupted turn to drain and the detector to resume',
      );
      await pumpFor(tester, const Duration(milliseconds: 1200));
      timeline.dispose();
      final barge = deps.diagnostics.latest.lastBargeIn!;
      debugPrint(
        'BARGE was_playing=${barge.wasPlaying} '
        'generating=$generatingAtBargeIn silenced=${ms(barge.silenced)}ms '
        'stop_confirmed=${ms(barge.stopConfirmed)}ms '
        'drain=${ms(barge.interruptDone)}ms unfrozen=${vm.frozen.value == null} '
        'recover=${timeline.recover}ms frames_1s=${timeline.framesFirstSecond}',
      );
      expect(barge.wasPlaying, isTrue);
      expect(
        barge.silenced,
        lessThanOrEqualTo(const Duration(milliseconds: 150)),
      );
      if (timeline.resumedAt != null) {
        expect(timeline.recover, lessThanOrEqualTo(1000));
      }
    }

    // 8 (before leaving Demo 3: moonshine is active). The router on real
    // moonshine transcripts of the golden questions.
    if (kRouterAudioDir.isNotEmpty) {
      final dir = Directory(kRouterAudioDir);
      final doc = jsonDecode(
        File('${dir.path}/index.json').readAsStringSync(),
      ) as Map<String, Object?>;
      final items = (doc['items']! as List<Object?>)
          .cast<Map<String, Object?>>();
      // The app's loaded recognizer (a singleton keyed by the model).
      final recognizer = await getActiveSttFor(kMoonshineSttConfig);
      const router = QuestionRouter();
      final misses = <String>[];
      var mustTotal = 0;
      var mustRight = 0;
      final watch = Stopwatch()..start();
      for (var i = 0; i < items.length; i++) {
        final item = items[i];
        final pcm = File('${dir.path}/$i.pcm').readAsBytesSync();
        final transcript = await recognizer.transcribe(pcm);
        final route = router.classify(transcript);
        final ok = switch ((item['route'], route)) {
          ('fast', FastRoute(:final intent, :final cls)) =>
            intent.name == item['intent'] &&
                (item['class'] == null ||
                    cls == kCocoNames.indexOf(item['class']! as String)),
          ('detailed', DetailedRoute()) => true,
          _ => false,
        };
        if (!ok) {
          misses.add(
            '"${(item['q']! as String).trim()}" heard "${transcript.trim()}" '
            '→ $route',
          );
        }
        if (item['must'] == true) {
          mustTotal++;
          if (ok) mustRight++;
        }
        await tester.pump();
      }
      final right = items.length - misses.length;
      debugPrint(
        'ROUTER_REAL stt=moonshine golden=${items.length} correct=$right '
        'accuracy=${(100 * right / items.length).toStringAsFixed(1)}% '
        'must_detailed=$mustRight/$mustTotal '
        'stt_total=${watch.elapsed.inMilliseconds}ms'
        '${misses.isEmpty ? '' : '\n  ${misses.join('\n  ')}'}',
      );
      expect(mustRight, mustTotal, reason: misses.join('\n'));
      expect(right / items.length, greaterThanOrEqualTo(0.95));
    }

    // 7. Back to Demo 1: Whisper again, a full voice turn.
    await backHome();
    await openDemo(Demo.voiceChat, VoiceChatScreen);
    final chat = Provider.of<VoiceChatViewModel>(
      tester.element(find.byType(VoiceChatScreen)),
      listen: false,
    );
    await pumpUntil(
      tester,
      () => chat.isReady && deps.activeStt.value?.id == ModelId.whisperBase,
      timeout: const Duration(seconds: 15),
      reason: 'Demo 1 open with Whisper active',
      describe: () =>
          'stt=${deps.activeStt.value?.modelId} '
          'error=${chat.error}',
    );
    final backToWhisper = deps.activeStt.value!.switchTime;
    TurnResult? result;
    unawaited(
      chat
          .submitUtterance(
            Utterance(pcm: france, held: pcm16Duration(france.length, 16000)),
          )
          .then((r) => result = r),
    );
    await pumpUntil(
      tester,
      () => result != null,
      timeout: const Duration(seconds: 90),
      reason: 'the Demo 1 voice turn',
      describe: () => 'phase=${chat.phase} error=${chat.error}',
    );
    expect(result!.outcome, TurnOutcome.completed);
    final user = chat.entries.lastWhere((e) => e.role == ChatRole.user);
    final reply = chat.entries.last;
    final voice = deps.diagnostics.latest.lastVoiceTurn!;
    debugPrint(
      'DEMO1 stt_model=${deps.activeStt.value?.modelId} '
      'switch=${ms(backToWhisper)}ms stt=${ms(voice.stt)}ms '
      'first_audio=${ms(voice.firstAudio)}ms transcript="${user.text}" '
      'reply="${reply.text}"',
    );
    expect(user.text.toLowerCase(), contains('france'));
    expect(reply.text.toLowerCase(), contains('paris'));

    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 45)));
}
