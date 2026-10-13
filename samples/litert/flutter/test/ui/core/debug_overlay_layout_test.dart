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

// The debug overlay on small and large windows: the panel stays inside the
// screen under the app bar, at most 440 wide and bounded in height with the
// demos' controls left free; on phones it starts collapsed to a summary, and
// its handle switches to the full list, which scrolls inside the panel. The
// panel lets pointers through to the screen (the home tiles open) except on
// the handle and on an expanded list that overflows.
//
// Text is laid out with the SDK's Roboto (standing in for the panel's Menlo
// too), not the test font's 1-em boxes, so wrapping and overflow are close
// to a device's.
//
// Screenshots, for a human look (skipped unless asked; Menlo on macOS):
//
//   fvm flutter test test/ui/core/debug_overlay_layout_test.dart \
//     --dart-define=DEBUG_OVERLAY_SHOTS=/some/dir
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';
import 'package:litert_edge_demos/ui/core/level_meter.dart';
import 'package:litert_edge_demos/ui/features/home/view_models/home_view_model.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/attachment_bar.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/composer.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/message_bubble.dart';
import 'package:provider/provider.dart';

import '../../fakes/fake_bundled_files.dart';
import '../../fakes/fake_detector_runtime.dart';
import '../../fakes/fake_knowledge.dart';
import '../../fakes/fake_llm_service.dart';
import '../../fakes/fake_model_files.dart';
import '../../fakes/fake_speech.dart';

/// Where the screenshots go; empty (the default) skips them.
const _shotsDir = String.fromEnvironment('DEBUG_OVERLAY_SHOTS');

final _now = DateTime(2026, 10, 7, 12);

Duration _ms(int ms) => Duration(milliseconds: ms);

LoadedModelInfo _model(
  String id,
  String backend, {
  String? detail,
  ChatModelFacts? chat,
}) => LoadedModelInfo(
  modelId: id,
  backend: backend,
  loadTime: _ms(612),
  warmUpTime: _ms(148),
  detail: detail,
  chat: chat,
);

/// Every section filled: live detection on a network camera, the last
/// generation with an image and tool calls, the knowledge base with a used
/// retrieval, a voice turn with a barge-in, and a detailed camera turn on a
/// frozen frame.
DiagnosticsSnapshot _longSnapshot() {
  final excerpts = [passage(1, similarity: 0.62), passage(2)];
  return DiagnosticsSnapshot(
    models: {
      ModelId.chat: LoadedModelInfo(
        modelId: 'gemma-4-E2B-it',
        backend: 'gpu',
        loadTime: _ms(4213),
        warmUpTime: _ms(812),
        chat: const ChatModelFacts(
          name: 'Gemma 4 E2B',
          custom: true,
          source: 'models folder',
          requestedBackend: 'gpu',
          requestedContext: 8192,
          contextTokens: 8192,
          images: true,
          tools: true,
          modelType: 'gemma4',
        ),
      ),
      ModelId.whisperBase: _model('whisper_base_int8', 'cpu'),
      ModelId.inflectNano: _model('inflect-nano-v2', 'cpu'),
      ModelId.yolo26n: _model(
        'yolo26n_fp16_rawhead',
        'gpu',
        detail: 'GPU fp32 full',
      ),
      ModelId.moonshineTiny: _model('moonshine_tiny_5s_f32', 'cpu'),
      ModelId.embeddingGemma: _model(
        'embeddinggemma-300M_seq512_mixed-precision',
        'cpu',
        detail: 'CPU · 768-d',
      ),
    },
    activeStt: ActiveSttInfo(
      id: ModelId.moonshineTiny,
      modelId: 'moonshine_tiny_5s_f32',
      switchTime: _ms(38),
    ),
    liveState: const LiveRunning('Network camera · 10.0.0.5:8080'),
    liveStats: const LiveStats(
      fps: 9.5,
      processed: 1200,
      sourceFrames: 1300,
      sourceFps: 9.5,
      sourceWidth: 1280,
      sourceHeight: 720,
      droppedBusy: 50,
      preMs: 12.4,
      runMs: 6.1,
      postMs: 0.9,
      latencyMs: 98.5,
    ),
    lastGeneration: GenerationMetrics(
      timeToFirstToken: _ms(2600),
      chunks: 42,
      tokensPerSecond: 21.5,
      tokensPerSecondSource: TokenRateSource.native,
      total: _ms(4600),
      stopped: false,
      imageAttached: true,
      imageSent: true,
      contextTokens: 2310,
      prefillTokens: 1650,
      promptTokens: 1360,
      toolRounds: 2,
      skillSteps: [
        SkillLoaded('device-info', found: true, at: _ms(840)),
        IntentCalled('device_info', '{}', at: _ms(1710)),
        IntentSucceeded(
          'device_info',
          'Apple M4 Pro',
          elapsed: _ms(2),
          at: _ms(1712),
        ),
      ],
    ),
    attachment: LlmImage(
      png: Uint8List(412000),
      width: 896,
      height: 672,
      sourceWidth: 4032,
      sourceHeight: 3024,
      sourceBytes: 2900000,
      normalizeTime: _ms(182),
    ),
    voicePhase: TurnPhase.speaking,
    lastVoiceTurn: VoiceTurnMetrics(
      typed: false,
      stt: _ms(65),
      firstAudio: _ms(1430),
      ttsClauses: [_ms(210), _ms(180), _ms(240)],
      outcome: TurnOutcome.completed,
      peakDbfs: -18.2,
      gateDbfs: -45,
      voiced: _ms(1820),
    ),
    lastBargeIn: BargeInMetrics(
      wasPlaying: true,
      silenced: _ms(42),
      interruptDone: _ms(310),
    ),
    audioDevices: const AudioDeviceStatus(
      input: DeviceReady('MacBook Pro Microphone'),
      output: DeviceReady('MacBook Pro Speakers'),
    ),
    skills: SkillCatalog(
      directory: '/skills',
      skills: [
        LoadedSkill(
          skill: parseSkillMd(
            '---\nname: device-info\ndescription: The device.\n---\nCall '
            'the `run_intent` tool with intent `device_info`.',
          ),
          path: 'device-info/SKILL.md',
        ),
      ],
      errors: const [
        SkillLoadError(path: 'broken/SKILL.md', message: 'no name'),
      ],
      fingerprint: 'f',
    ),
    knowledge: KnowledgeReady(
      chunks: 290,
      reused: true,
      elapsed: _ms(27),
      origin: KnowledgeOrigin.prebuilt,
    ),
    lastRetrieval: Retrieval(
      outcome: RetrievalOutcome.used,
      passages: excerpts,
      candidates: excerpts,
      gate: 0.4,
      latency: _ms(131),
    ),
    lastCameraTurn: CameraTurnMetrics(
      route: const DetailedRoute('describe'),
      routeTime: const Duration(microseconds: 420),
      snapshotLatency: _ms(96),
      image: EncodedSnapshot(
        png: Uint8List(380000),
        frameId: 1234,
        width: 896,
        height: 504,
        unmirrored: false,
        encodeTime: _ms(41),
      ),
      timeToFirstToken: _ms(1880),
    ),
    lastCameraReset: _ms(120),
    frozenFrameId: 1234,
    llmTurns: 7,
    rssBytes: 2254857830,
    peakRssBytes: 2791728742,
  );
}

const _bodyKey = ValueKey('under-the-panel');
const _micKey = ValueKey('mic');

/// Demo 1's layout around the host as the app puts it (MaterialApp.builder):
/// an app bar with the toggle, a message list, and the real bottom controls
/// (phase line, attachment bar, composer with the mic).
Widget _app(
  ValueListenable<DiagnosticsSnapshot> snapshot, {
  VoidCallback? onBodyTap,
  VoidCallback? onMic,
}) {
  const entries = [
    ChatEntry(role: ChatRole.user, text: 'Which accelerator are you on?'),
    ChatEntry(
      role: ChatRole.assistant,
      text:
          'I am running on the GPU of an Apple M4 Pro; the speech models run '
          'on the CPU.',
    ),
    ChatEntry(role: ChatRole.user, text: 'What is LiteRT?'),
    ChatEntry(
      role: ChatRole.assistant,
      text:
          "LiteRT is Google's on-device runtime for machine-learning models, "
          'formerly TensorFlow Lite [1].',
    ),
  ];
  return MaterialApp(
    debugShowCheckedModeBanner: false,
    builder: (context, child) =>
        DebugOverlayHost(snapshot: snapshot, child: child!),
    home: Scaffold(
      appBar: AppBar(
        title: const Text('Voice chat'),
        actions: const [DebugOverlayToggle()],
      ),
      body: Column(
        children: [
          Expanded(
            child: GestureDetector(
              key: _bodyKey,
              behavior: HitTestBehavior.opaque,
              onTap: onBodyTap,
              child: ListView(
                padding: const EdgeInsets.all(12),
                children: [
                  for (final entry in entries) MessageBubble(entry: entry),
                ],
              ),
            ),
          ),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            child: Text('Ready'),
          ),
          AttachmentBar(
            attachment: null,
            picking: false,
            showGallery: true,
            showCamera: true,
            onGallery: () {},
            onCamera: () {},
            onRemove: () {},
          ),
          Composer(
            canSend: true,
            isGenerating: false,
            onSend: (_) {},
            onStop: () {},
            leading: MicButton(
              key: _micKey,
              level: const AlwaysStoppedAnimation(0),
              phase: TurnPhase.idle,
              enabled: true,
              onDown: onMic ?? () {},
              onUp: () {},
            ),
          ),
        ],
      ),
    ),
  );
}

/// A window of [size] logical pixels with [insets] (status bar, notch,
/// home indicator), at device pixel ratio 1.
void _window(WidgetTester tester, Size size, EdgeInsets insets) {
  tester.view.devicePixelRatio = 1;
  tester.view.physicalSize = size;
  final padding = FakeViewPadding(
    left: insets.left,
    top: insets.top,
    right: insets.right,
    bottom: insets.bottom,
  );
  tester.view.padding = padding;
  tester.view.viewPadding = padding;
  addTearDown(tester.view.reset);
}

/// A device: its window, its insets and the panel's share of the safe
/// area's height.
typedef _Device = ({String name, Size size, EdgeInsets insets, double share});

const _phone = (
  name: 'phone',
  size: Size(360, 780),
  insets: EdgeInsets.only(top: 24, bottom: 16),
  share: 0.45,
);
const _largePhone = (
  name: 'large phone',
  size: Size(412, 915),
  insets: EdgeInsets.only(top: 47, bottom: 34),
  share: 0.45,
);
const _desktop = (
  name: 'desktop',
  size: Size(1280, 800),
  insets: EdgeInsets.zero,
  share: 0.75,
);
const _landscapePhone = (
  name: 'landscape phone',
  size: Size(780, 360),
  insets: EdgeInsets.only(left: 47, right: 47, bottom: 21),
  share: 0.45,
);

Rect _panel(WidgetTester tester) =>
    tester.getRect(find.byKey(DebugOverlayKeys.panel));

ScrollPosition _scroll(WidgetTester tester) => tester
    .state<ScrollableState>(
      find.descendant(
        of: find.byKey(DebugOverlayKeys.scroll),
        matching: find.byType(Scrollable),
      ),
    )
    .position;

/// A panel line containing [s]; the panel keeps numbers with their units
/// with no-break spaces, read here as spaces.
Finder _text(String s) => find.byWidgetPredicate(
  (w) => w is Text && (w.data ?? '').replaceAll('\u00a0', ' ').contains(s),
  description: 'a Text containing "$s"',
);

/// The handle, the panel's only switch.
Finder get _handle => find.byKey(DebugOverlayKeys.expand);

/// Expanded: the handle offers `less`.
bool _expanded() => find
    .descendant(of: _handle, matching: find.byIcon(Icons.unfold_less))
    .evaluate()
    .isNotEmpty;

/// Taps the handle and lets the panel settle: the new text is laid out, its
/// overflow measured, the body's pointer mode updated, the thumb faded.
Future<void> _tapHandle(WidgetTester tester) async {
  await tester.tap(_handle);
  await tester.pumpAndSettle();
}

/// The scroll strip, present only while the text overflows.
Finder get _strip => find.byKey(DebugOverlayKeys.scrollbar);

/// The panel lies inside the safe area, under the app bar, at most 440
/// wide, within its height share, with the bottom controls left free; its
/// handle (at least 32 dp high) and its scroll strip, when there is one,
/// lie inside it and cover no text.
void _expectBounded(WidgetTester tester, _Device device) {
  expect(tester.takeException(), isNull);
  final screen = Offset.zero & device.size;
  final safe = device.insets.deflateRect(screen);
  final panel = _panel(tester);
  final appBar = tester.getRect(find.byType(AppBar));
  final reason = '${device.name}: panel $panel in safe area $safe';
  expect(panel.left, greaterThanOrEqualTo(safe.left + 8), reason: reason);
  expect(panel.right, moreOrLessEquals(safe.right - 8), reason: reason);
  expect(panel.top, greaterThan(appBar.bottom), reason: reason);
  expect(panel.width, lessThanOrEqualTo(440), reason: reason);
  expect(
    panel.height,
    lessThanOrEqualTo(device.share * safe.height + 0.01),
    reason: reason,
  );
  expect(panel.bottom, lessThanOrEqualTo(safe.bottom - 160), reason: reason);
  // The mic and the composer stay clear of the panel.
  final mic = tester.getRect(find.byKey(_micKey));
  expect(panel.overlaps(mic), isFalse, reason: '$reason, mic $mic');
  expect(
    panel.overlaps(tester.getRect(find.byType(Composer))),
    isFalse,
    reason: reason,
  );
  // The handle: inside the panel, a finger-sized target, over no text (each
  // line as far as the scroll view shows it).
  final handle = tester.getRect(_handle);
  expect(panel.inflate(0.01).contains(handle.topLeft), isTrue, reason: reason);
  expect(
    panel.inflate(0.01).contains(handle.bottomRight),
    isTrue,
    reason: reason,
  );
  expect(handle.height, greaterThanOrEqualTo(32), reason: reason);
  final targets = [handle];
  if (_strip.evaluate().isNotEmpty) {
    final strip = tester.getRect(_strip);
    expect(panel.inflate(0.01).contains(strip.topLeft), isTrue);
    expect(panel.inflate(0.01).contains(strip.bottomRight), isTrue);
    targets.add(strip);
  }
  final viewport = tester.getRect(find.byKey(DebugOverlayKeys.scroll));
  for (final text
      in find
          .descendant(
            of: find.byKey(DebugOverlayKeys.scroll),
            matching: find.byType(Text),
          )
          .evaluate()) {
    final shown = tester
        .getRect(find.byWidget(text.widget))
        .intersect(viewport);
    if (shown.isEmpty) continue;
    for (final target in targets) {
      expect(target.overlaps(shown), isFalse, reason: '$reason, text $shown');
    }
  }
}

/// After both demos: everything but live detection, which stopped when
/// Demo 3 closed (about 25 lines).
DiagnosticsSnapshot _postDemoSnapshot() =>
    _longSnapshot().copyWith(liveState: const LiveStopped());

const _macWindow = (
  name: 'macOS window',
  size: Size(800, 600),
  insets: EdgeInsets.zero,
  share: 0.75,
);

/// Right after a fresh launch: every model loaded, the knowledge base and
/// the skills ready, nothing run yet.
DiagnosticsSnapshot _homeSnapshot() {
  final s = _longSnapshot();
  return DiagnosticsSnapshot(
    models: s.models,
    activeStt: const ActiveSttInfo(
      id: ModelId.whisperBase,
      modelId: 'whisper_base_int8',
    ),
    knowledge: s.knowledge,
    skills: s.skills,
    rssBytes: 1534000000,
    peakRssBytes: 2233000000,
  );
}

/// The app's models with every service faked, both demos available.
Future<ModelRepository> _homeModels(WidgetTester tester) async {
  final models = ModelRepository(
    gemmaModelPath: kTestChatModelPath,
    bundled: FakeBundledFiles(),
    detector: fakeDetectorService(),
    bundledDetector: fakeBundledDetector,
    llm: FakeLlmService(),
    stt: fakeSttService(),
    tts: fakeTtsService(),
    embedder: EmbedderService(),
  );
  await tester.runAsync(models.prepareAll);
  return models;
}

/// The real home screen under the host, as the app builds it.
Widget _home(
  ValueListenable<DiagnosticsSnapshot> snapshot,
  ModelRepository models,
  List<Demo> opened,
) => MaterialApp(
  debugShowCheckedModeBanner: false,
  builder: (context, child) =>
      DebugOverlayHost(snapshot: snapshot, child: child!),
  home: ChangeNotifierProvider(
    create: (_) => HomeViewModel(models: models.states),
    child: HomeScreen(onOpen: (demo) async => opened.add(demo)),
  ),
);

void main() {
  setUpAll(_loadSdkFonts);

  group('debugOverlaySummaryLines', () {
    test('the chat backend, the detector fps and latency, TTFT and RSS in '
        'a few lines, the same text as the full list', () {
      final s = _longSnapshot();
      final full = debugOverlayLines(s, now: _now);
      final summary = debugOverlaySummaryLines(s, now: _now);

      expect(summary, [
        'Gemma 4 E2B · GPU · load 4213 ms · warm 812 ms',
        'det GPU fp32 full · 9.5 fps · lat 98.5 ms',
        'TTFT 2600 ms · 21.5 tok/s (native) · chunks 42',
        'RSS 2.10 GB · peak 2.60 GB',
        '+${full.length - 4} more',
      ]);
      expect(full, containsAll([summary[0], summary[2], summary[3]]));
      // The full list's detector lines carry the same figures.
      expect(full, contains(startsWith('det GPU fp32 full · 9.5 fps · ')));
      expect(full, contains(contains(' · lat 98.5 ms · ')));
    });

    test('the detector line follows the live state; none while stopped', () {
      final s = _longSnapshot();
      List<String> summary(LiveState state) =>
          debugOverlaySummaryLines(s.copyWith(liveState: state), now: _now);

      expect(
        summary(const LiveStopped()).where((l) => l.startsWith('det')),
        isEmpty,
      );
      expect(summary(const LiveStarting())[1], 'det GPU fp32 full · starting…');
      expect(
        summary(
          LivePaused(
            source: 'camera',
            reason: 'gemma',
            since: _now.subtract(_ms(1200)),
          ),
        )[1],
        'det paused (gemma) 1.2 s',
      );
      expect(
        summary(const LiveFailed('no frames'))[1],
        'det failed: no frames',
      );
    });

    test('nothing loaded yet', () {
      expect(debugOverlaySummaryLines(const DiagnosticsSnapshot()), [
        'Chat model: not loaded',
        'TTFT – · tok/s –',
        'RSS – · peak –',
        '+${debugOverlayLines(const DiagnosticsSnapshot()).length - 3} more',
      ]);
    });
  });

  group('the panel fits the window', () {
    for (final device in [_phone, _largePhone, _desktop, _landscapePhone]) {
      testWidgets('${device.name} ${device.size}: inside the screen under '
          'the app bar, ≤ 440 wide, bounded, collapsed and expanded', (
        tester,
      ) async {
        _window(tester, device.size, device.insets);
        final snapshot = ValueNotifier(_longSnapshot());
        addTearDown(snapshot.dispose);
        await tester.pumpWidget(_app(snapshot));
        await tester.pumpAndSettle();

        _expectBounded(tester, device);
        await _tapHandle(tester);
        _expectBounded(tester, device);
        await _tapHandle(tester);
        _expectBounded(tester, device);
      });
    }
  });

  testWidgets('phone: the text lets taps through, collapsed or expanded; '
      'the handle expands the full list, which overflows and scrolls by its '
      'strip (drag, wheel); the handle collapses it', (tester) async {
    _window(tester, _phone.size, _phone.insets);
    final snapshot = ValueNotifier(_longSnapshot());
    addTearDown(snapshot.dispose);
    var bodyTaps = 0;
    var micPresses = 0;
    await tester.pumpWidget(
      _app(snapshot, onBodyTap: () => bodyTaps++, onMic: () => micPresses++),
    );
    await tester.pumpAndSettle();
    final more = debugOverlayLines(_longSnapshot(), now: _now).length - 4;

    // Collapsed: the essentials, and the handle to the rest.
    expect(_expanded(), isFalse);
    expect(
      find.descendant(of: _handle, matching: find.text('+$more more')),
      findsOneWidget,
    );
    expect(_text('det GPU fp32 full · 9.5 fps · lat 98.5 ms'), findsOneWidget);
    expect(_text('det src'), findsNothing);
    // A wrap never splits a number from its unit.
    expect(
      find.textContaining('9.5\u00a0fps · lat 98.5\u00a0ms'),
      findsOneWidget,
    );
    final collapsed = _panel(tester);
    expect(_strip, findsNothing, reason: 'the summary fits');

    // On the panel's text, under it and beside it: the screen gets the tap.
    await tester.tapAt(collapsed.center);
    expect(bodyTaps, 1, reason: 'the collapsed body lets the tap through');
    expect(_expanded(), isFalse, reason: 'only the handle toggles');
    await tester.tapAt(Offset(collapsed.center.dx, collapsed.bottom + 24));
    await tester.tapAt(Offset(4, collapsed.center.dy));
    expect(bodyTaps, 3);
    await tester.tap(find.byKey(_micKey));
    expect(micPresses, 1);

    // Expanded: the full list, taller than the bound.
    await _tapHandle(tester);
    expect(_expanded(), isTrue);
    expect(_text('det src Network camera'), findsOneWidget);
    expect(
      find.descendant(of: _handle, matching: find.text('less')),
      findsOneWidget,
    );
    final expanded = _panel(tester);
    expect(expanded.height, greaterThan(collapsed.height));
    expect(_scroll(tester).maxScrollExtent, greaterThan(0));

    // It overflows, and the text still lets taps through; the strip on the
    // right edge is what scrolls.
    await tester.tapAt(expanded.center);
    expect(bodyTaps, 4, reason: 'the overflowing text lets the tap through');
    expect(_expanded(), isTrue);
    expect(_strip, findsOneWidget);
    final strip = tester.getRect(_strip);
    expect(strip.width, 24);
    expect(strip.right, moreOrLessEquals(expanded.right));

    // A tap on the strip neither toggles nor reaches the screen.
    await tester.tapAt(strip.center);
    await tester.pump();
    expect(_expanded(), isTrue);
    expect(bodyTaps, 4);

    // A drag along the strip moves the thumb (down: later lines).
    await tester.drag(_strip, const Offset(0, 40));
    await tester.pumpAndSettle();
    final dragged = _scroll(tester).pixels;
    expect(dragged, greaterThan(0));
    expect(_expanded(), isTrue, reason: 'a drag scrolls, it does not toggle');
    expect(bodyTaps, 4, reason: 'the drag stayed on the strip');

    // The mouse wheel over the strip scrolls too (desktop).
    final mouse = TestPointer(1, PointerDeviceKind.mouse);
    await tester.sendEventToBinding(mouse.hover(strip.center));
    await tester.sendEventToBinding(mouse.scroll(const Offset(0, 60)));
    await tester.pump();
    expect(_scroll(tester).pixels, greaterThan(dragged));

    // Below the expanded panel the screen still gets its taps.
    await tester.tapAt(Offset(expanded.center.dx, expanded.bottom + 24));
    expect(bodyTaps, 5);

    // Collapsed again, from the top, without the strip.
    await _tapHandle(tester);
    expect(_expanded(), isFalse);
    expect(_scroll(tester).pixels, 0);
    expect(_strip, findsNothing);
    await tester.tapAt(_panel(tester).center);
    expect(bodyTaps, 6);

    // The app-bar toggle still hides and shows the panel.
    await tester.tap(find.byKey(DebugOverlayKeys.toggle));
    await tester.pump();
    expect(find.byKey(DebugOverlayKeys.panel), findsNothing);
    await tester.tap(find.byKey(DebugOverlayKeys.toggle));
    await tester.pump();
    expect(find.byKey(DebugOverlayKeys.panel), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('desktop: expanded by default; the list fits, so no strip; '
      'the text lets taps through; the handle collapses it', (tester) async {
    _window(tester, _desktop.size, _desktop.insets);
    final snapshot = ValueNotifier(_longSnapshot());
    addTearDown(snapshot.dispose);
    var bodyTaps = 0;
    await tester.pumpWidget(_app(snapshot, onBodyTap: () => bodyTaps++));
    await tester.pumpAndSettle();

    expect(_expanded(), isTrue);
    expect(_text('det src Network camera'), findsOneWidget);
    expect(_panel(tester).width, 440);
    expect(_scroll(tester).maxScrollExtent, 0, reason: 'it all fits');
    expect(_strip, findsNothing);
    await tester.tapAt(_panel(tester).center);
    expect(bodyTaps, 1, reason: 'a list that fits lets the tap through');
    expect(_expanded(), isTrue);

    await _tapHandle(tester);
    expect(_expanded(), isFalse);
    expect(bodyTaps, 1, reason: 'the handle keeps its tap');
  });

  group('the real home screen under the panel: both tiles open', () {
    for (final (name, device, snapshotOf, expand, overflows) in [
      ('phone, collapsed, at launch', _phone, _homeSnapshot, false, false),
      (
        'the macOS window, expanded, at launch',
        _macWindow,
        _homeSnapshot,
        false,
        false,
      ),
      (
        // camera_assistant_test goes home after Demo 3 and taps Demo 1.
        'the macOS window, expanded, after the demos',
        _macWindow,
        _postDemoSnapshot,
        false,
        true,
      ),
      (
        'phone, expanded, after the demos',
        _phone,
        _postDemoSnapshot,
        true,
        true,
      ),
    ]) {
      testWidgets('$name ${device.size}', (tester) async {
        _window(tester, device.size, device.insets);
        final snapshot = ValueNotifier(snapshotOf());
        addTearDown(snapshot.dispose);
        final models = await _homeModels(tester);
        final opened = <Demo>[];
        await tester.pumpWidget(_home(snapshot, models, opened));
        await tester.pumpAndSettle();
        if (expand) await _tapHandle(tester);

        expect(_expanded(), device.size.width >= 600 || expand);
        expect(_scroll(tester).maxScrollExtent, overflows ? greaterThan(0) : 0);
        expect(_strip, overflows ? findsOneWidget : findsNothing);
        final panel = _panel(tester);
        final chat = tester.getCenter(
          find.byKey(HomeKeys.tile(Demo.voiceChat)),
        );
        // The case that matters: Demo 1's tile is under the panel, clear of
        // the handle and the strip.
        expect(panel.contains(chat), isTrue);
        expect(tester.getRect(_handle).contains(chat), isFalse);
        if (overflows) {
          expect(tester.getRect(_strip).contains(chat), isFalse);
        }
        for (final demo in Demo.values) {
          final tile = find.byKey(HomeKeys.tile(demo));
          expect(tester.widget<ListTile>(tile).enabled, isTrue);
          await tester.tap(tile);
          await tester.pump();
        }
        expect(opened, Demo.values);
        expect(tester.takeException(), isNull);

        await tester.pumpWidget(const SizedBox.shrink());
        await tester.runAsync(models.close);
      });
    }
  });

  testWidgets('phone with the keyboard open: the expanded panel shrinks '
      'above it, clear of the composer', (tester) async {
    _window(tester, _phone.size, const EdgeInsets.only(top: 24));
    tester.view.viewInsets = const FakeViewPadding(bottom: 300);
    final snapshot = ValueNotifier(_longSnapshot());
    addTearDown(snapshot.dispose);
    await tester.pumpWidget(_app(snapshot));
    await tester.pumpAndSettle();
    await _tapHandle(tester);

    expect(tester.takeException(), isNull);
    expect(_expanded(), isTrue);
    final panel = _panel(tester);
    final aboveKeyboard = _phone.size.height - 300;
    expect(panel.bottom, lessThanOrEqualTo(aboveKeyboard - 160));
    expect(panel.height, lessThanOrEqualTo(0.45 * (aboveKeyboard - 24) + 0.01));
    expect(panel.overlaps(tester.getRect(find.byType(Composer))), isFalse);
  });

  testWidgets('screen readers: the handle is its own button with a tap '
      'action, the strip scrolls', (tester) async {
    final semantics = tester.ensureSemantics();
    _window(tester, _phone.size, _phone.insets);
    final snapshot = ValueNotifier(_longSnapshot());
    addTearDown(snapshot.dispose);
    await tester.pumpWidget(_app(snapshot));
    await tester.pumpAndSettle();

    final handle = tester.getSemantics(_handle);
    expect(
      handle,
      isSemantics(
        label: 'Show all diagnostics',
        isButton: true,
        hasTapAction: true,
      ),
    );
    expect(handle.rect.size, tester.getSize(_handle), reason: 'not the screen');
    handle.owner!.performAction(handle.id, SemanticsAction.tap);
    await tester.pumpAndSettle();
    expect(_expanded(), isTrue);

    final strip = tester.getSemantics(_strip);
    expect(
      strip,
      isSemantics(
        label: 'Diagnostics scroll bar',
        hasScrollUpAction: true,
        hasScrollDownAction: true,
      ),
    );
    expect(strip.rect.size, tester.getSize(_strip));
    strip.owner!.performAction(strip.id, SemanticsAction.scrollUp);
    await tester.pump();
    expect(_scroll(tester).pixels, greaterThan(0), reason: 'later lines');
    semantics.dispose();
  });

  testWidgets('a window too short or too narrow for a line and the handle '
      'gets no panel, and no layout error', (tester) async {
    final snapshot = ValueNotifier(_longSnapshot());
    addTearDown(snapshot.dispose);
    Widget app() => MaterialApp(
      builder: (context, child) =>
          DebugOverlayHost(snapshot: snapshot, child: child!),
      home: Scaffold(
        appBar: AppBar(actions: const [DebugOverlayToggle()]),
        body: const SizedBox.expand(),
      ),
    );

    for (final size in const [Size(800, 80), Size(800, 120), Size(150, 600)]) {
      _window(tester, size, EdgeInsets.zero);
      await tester.pumpWidget(app());
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull, reason: '$size');
      expect(find.byKey(DebugOverlayKeys.panel), findsNothing, reason: '$size');
    }
    // Just tall enough: a line and the handle, no overflow.
    _window(tester, const Size(800, 124), EdgeInsets.zero);
    await tester.pumpWidget(app());
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    expect(find.byKey(DebugOverlayKeys.panel), findsOneWidget);
  });

  testWidgets('the expand choice survives hiding the panel', (tester) async {
    _window(tester, _phone.size, _phone.insets);
    final snapshot = ValueNotifier(_longSnapshot());
    addTearDown(snapshot.dispose);
    await tester.pumpWidget(_app(snapshot));
    await tester.pumpAndSettle();
    await _tapHandle(tester);
    expect(_expanded(), isTrue);

    await tester.tap(find.byKey(DebugOverlayKeys.toggle));
    await tester.pumpAndSettle();
    expect(find.byKey(DebugOverlayKeys.panel), findsNothing);
    await tester.tap(find.byKey(DebugOverlayKeys.toggle));
    await tester.pumpAndSettle();
    expect(_expanded(), isTrue);
  });

  testWidgets('the panel follows the snapshot', (tester) async {
    _window(tester, _phone.size, _phone.insets);
    final snapshot = ValueNotifier(_longSnapshot());
    addTearDown(snapshot.dispose);
    await tester.pumpWidget(_app(snapshot));

    snapshot.value = snapshot.value.copyWith(
      liveStats: const LiveStats(fps: 29.7, latencyMs: 6),
    );
    await tester.pump();
    expect(_text('det GPU fp32 full · 29.7 fps · lat 6.0 ms'), findsOneWidget);
  });

  group('screenshots', () {
    final boundary = GlobalKey();

    Future<void> shot(WidgetTester tester, String name, double ratio) async {
      await tester.pumpAndSettle();
      final render =
          boundary.currentContext!.findRenderObject()! as RenderRepaintBoundary;
      final bytes = await tester.runAsync(() async {
        final image = await render.toImage(pixelRatio: ratio);
        final data = await image.toByteData(format: ui.ImageByteFormat.png);
        image.dispose();
        return data!.buffer.asUint8List();
      });
      final file = File('$_shotsDir/$name.png');
      file.parent.createSync(recursive: true);
      file.writeAsBytesSync(bytes!);
      debugPrint('DEBUG_OVERLAY_SHOT ${file.path}');
    }

    /// A fresh tree on [device]: the last device's handle does not carry
    /// over.
    Future<void> pump(WidgetTester tester, _Device device, Widget app) async {
      _window(tester, device.size, device.insets);
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pumpWidget(RepaintBoundary(key: boundary, child: app));
    }

    testWidgets('home and chat on a phone, collapsed and expanded; large '
        'phone; landscape; desktop and the macOS window', (tester) async {
      final chat = ValueNotifier(_longSnapshot());
      addTearDown(chat.dispose);
      final home = ValueNotifier(_homeSnapshot());
      addTearDown(home.dispose);
      final models = await _homeModels(tester);
      final opened = <Demo>[];
      final afterDemos = ValueNotifier(_postDemoSnapshot());
      addTearDown(afterDemos.dispose);

      await pump(tester, _phone, _home(home, models, opened));
      await shot(tester, 'phone_home_collapsed', 3);
      await _tapHandle(tester);
      await shot(tester, 'phone_home_expanded', 3);

      await pump(tester, _phone, _app(chat));
      await shot(tester, 'phone_chat_collapsed', 3);
      await _tapHandle(tester);
      await shot(tester, 'phone_chat_expanded', 3);

      await pump(tester, _largePhone, _app(chat));
      await _tapHandle(tester);
      await shot(tester, 'large_phone_412x915_chat_expanded', 3);

      await pump(tester, _landscapePhone, _app(chat));
      await shot(tester, 'landscape_phone_780x360_chat_collapsed', 3);

      await pump(tester, _desktop, _app(chat));
      await shot(tester, 'desktop_1280x800_chat_expanded', 1.5);

      await pump(tester, _macWindow, _home(home, models, opened));
      await shot(tester, 'macos_window_800x600_home_expanded', 2);

      await pump(tester, _macWindow, _home(afterDemos, models, opened));
      await shot(tester, 'macos_window_800x600_home_after_demos', 2);

      await tester.pumpWidget(const SizedBox.shrink());
      await tester.runAsync(models.close);
    }, skip: _shotsDir.isEmpty);
  });
}

/// The Flutter SDK's fonts, where flutter_tester runs:
/// `bin/cache/artifacts/material_fonts`.
Directory get _sdkFonts => Directory(
  '${File(Platform.resolvedExecutable).parent.parent.parent.path}/'
  'material_fonts',
);

Future<void> _loadFont(String family, String path) async {
  final bytes = File(path).readAsBytesSync();
  await (FontLoader(
    family,
  )..addFont(Future.value(ByteData.sublistView(bytes)))).load();
}

/// Real glyphs instead of the test font's boxes: Roboto (the Material
/// theme's font) and the Material icons, and for the panel's `Menlo` the
/// real one for screenshots on macOS, else Roboto under that name. (The
/// panel's `fontFamilyFallback` never applies here: the test engine draws
/// an unknown family with its own font, which has every glyph.)
Future<void> _loadSdkFonts() async {
  final fonts = _sdkFonts;
  if (!fonts.existsSync()) {
    throw StateError('no Material fonts at ${fonts.path}');
  }
  final roboto = '${fonts.path}/Roboto-Regular.ttf';
  await _loadFont('Roboto', roboto);
  await _loadFont('MaterialIcons', '${fonts.path}/MaterialIcons-Regular.otf');
  const menlo = '/System/Library/Fonts/Menlo.ttc';
  await _loadFont(
    'Menlo',
    _shotsDir.isNotEmpty && File(menlo).existsSync() ? menlo : roboto,
  );
}
