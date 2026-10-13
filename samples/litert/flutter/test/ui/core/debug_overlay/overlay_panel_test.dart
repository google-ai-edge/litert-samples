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

// The DebugOverlay panel on its own, outside the host: which view it starts
// in, how the caller's expand choice drives it, and how it keeps numbers
// with their units. Layout, pointers and scrolling under a real screen are
// in debug_overlay_layout_test.dart.
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';

/// The caller's expand choice, saying whether the panel still listens.
final class _Choice extends ValueNotifier<bool?> {
  _Choice({bool? initial}) : super(initial);

  bool get listened => hasListeners;
}

const _empty = DiagnosticsSnapshot();

Widget _panel(
  ValueListenable<DiagnosticsSnapshot> snapshot,
  ValueNotifier<bool?> choice, {
  bool compact = false,
}) => MaterialApp(
  home: Material(
    child: Align(
      alignment: Alignment.topRight,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 440, maxHeight: 560),
        child: DebugOverlay(
          snapshot: snapshot,
          expandedChoice: choice,
          compact: compact,
        ),
      ),
    ),
  ),
);

Finder get _handle => find.byKey(DebugOverlayKeys.expand);

/// The handle offers `less`: the full list shows.
bool _expanded() => find
    .descendant(of: _handle, matching: find.byIcon(Icons.unfold_less))
    .evaluate()
    .isNotEmpty;

String _handleLabel(WidgetTester tester) => tester
    .widget<Text>(find.descendant(of: _handle, matching: find.byType(Text)))
    .data!;

/// The facts on the panel, in order.
List<String> _shown(WidgetTester tester) => [
  for (final text in tester.widgetList<Text>(
    find.descendant(
      of: find.byKey(DebugOverlayKeys.scroll),
      matching: find.byType(Text),
    ),
  ))
    text.data!,
];

double? _fontSize(WidgetTester tester) => tester
    .renderObject<RenderParagraph>(
      find
          .descendant(
            of: find.byKey(DebugOverlayKeys.scroll),
            matching: find.byType(RichText),
          )
          .first,
    )
    .text
    .style
    ?.fontSize;

void main() {
  testWidgets('a large window with no choice: the full list, "less" on the '
      'handle, 11 px text', (tester) async {
    final snapshot = ValueNotifier(_empty);
    addTearDown(snapshot.dispose);
    final choice = _Choice();
    addTearDown(choice.dispose);

    await tester.pumpWidget(_panel(snapshot, choice));

    expect(_expanded(), isTrue);
    expect(_shown(tester), debugOverlayLines(_empty));
    expect(_handleLabel(tester), 'less');
    expect(_fontSize(tester), 11);
    expect(choice.value, isNull, reason: 'no choice made by showing it');
  });

  testWidgets('a phone with no choice: the summary, the count of the rest on '
      'the handle, 10 px text', (tester) async {
    final snapshot = ValueNotifier(_empty);
    addTearDown(snapshot.dispose);
    final choice = _Choice();
    addTearDown(choice.dispose);

    await tester.pumpWidget(_panel(snapshot, choice, compact: true));

    final summary = debugOverlaySummaryLines(_empty);
    expect(_expanded(), isFalse);
    expect(_shown(tester), summary.sublist(0, summary.length - 1));
    expect(_handleLabel(tester), summary.last);
    expect(_handleLabel(tester), matches(RegExp(r'^\+\d+ more$')));
    expect(_fontSize(tester), 10);
  });

  testWidgets('with no choice the panel follows the window across the '
      'breakpoint; a tap on the handle is a choice the window no longer '
      'overrides', (tester) async {
    final snapshot = ValueNotifier(_empty);
    addTearDown(snapshot.dispose);
    final choice = _Choice();
    addTearDown(choice.dispose);

    await tester.pumpWidget(_panel(snapshot, choice, compact: true));
    expect(_expanded(), isFalse);
    await tester.pumpWidget(_panel(snapshot, choice));
    expect(_expanded(), isTrue, reason: 'the window grew');
    await tester.pumpWidget(_panel(snapshot, choice, compact: true));
    expect(_expanded(), isFalse, reason: 'and shrank again');

    await tester.tap(_handle);
    await tester.pump();
    expect(_expanded(), isTrue);
    expect(choice.value, isTrue, reason: "kept by the caller, not the panel");

    await tester.pumpWidget(_panel(snapshot, choice));
    await tester.pumpWidget(_panel(snapshot, choice, compact: true));
    expect(_expanded(), isTrue, reason: 'the choice outlives the resize');

    await tester.tap(_handle);
    await tester.pump();
    expect(_expanded(), isFalse);
    expect(choice.value, isFalse);
  });

  testWidgets("the choice is the caller's: a change from outside shows at "
      'once; a new notifier replaces the old one, which is let go', (
    tester,
  ) async {
    final snapshot = ValueNotifier(_empty);
    addTearDown(snapshot.dispose);
    final first = _Choice();
    addTearDown(first.dispose);
    final second = _Choice(initial: false);
    addTearDown(second.dispose);

    await tester.pumpWidget(_panel(snapshot, first));
    expect(_expanded(), isTrue);
    first.value = false;
    await tester.pump();
    expect(_expanded(), isFalse);

    await tester.pumpWidget(_panel(snapshot, second));
    expect(_expanded(), isFalse);
    expect(first.listened, isFalse);
    expect(second.listened, isTrue);
    first.value = true;
    await tester.pump();
    expect(_expanded(), isFalse, reason: 'the old notifier is not followed');
    second.value = true;
    await tester.pump();
    expect(_expanded(), isTrue);

    await tester.pumpWidget(const SizedBox.shrink());
    expect(second.listened, isFalse, reason: 'disposed with the panel');
  });

  testWidgets('a number keeps its unit: a no-break space before ms, s, fps, '
      'tok, KB, MB, GB and dBFS, so a wrap never splits them', (tester) async {
    final snapshot = ValueNotifier(
      DiagnosticsSnapshot(
        models: {
          ModelId.chat: const LoadedModelInfo(
            modelId: 'gemma',
            backend: 'gpu',
            loadTime: Duration(milliseconds: 612),
            warmUpTime: Duration(milliseconds: 148),
          ),
        },
        liveState: const LiveRunning('camera'),
        liveStats: const LiveStats(fps: 15),
        lastGeneration: const GenerationMetrics(
          timeToFirstToken: Duration(milliseconds: 2600),
          chunks: 42,
          tokensPerSecond: 21.5,
          tokensPerSecondSource: TokenRateSource.native,
          total: Duration(seconds: 4),
          stopped: false,
          imageAttached: true,
          imageSent: true,
          contextTokens: 2310,
          prefillTokens: 1650,
          promptTokens: 1360,
          toolRounds: 1,
          skillSteps: [
            SkillLoaded(
              'device-info',
              found: true,
              at: Duration(milliseconds: 840),
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
          normalizeTime: const Duration(milliseconds: 182),
        ),
        lastVoiceTurn: const VoiceTurnMetrics(
          typed: false,
          stt: Duration(milliseconds: 65),
          peakDbfs: -18.2,
          gateDbfs: -45,
          voiced: Duration(milliseconds: 1820),
        ),
        rssBytes: 700 << 20,
        peakRssBytes: 3 << 29,
      ),
    );
    addTearDown(snapshot.dispose);
    final choice = _Choice(initial: true);
    addTearDown(choice.dispose);

    await tester.pumpWidget(_panel(snapshot, choice));

    final shown = _shown(tester);
    final text = shown.join('\n');
    for (final kept in [
      '612\u00A0ms',
      '0.84\u00A0s',
      '15.0\u00A0fps',
      '2310/${kLlmConfig.maxTokens}\u00A0tok',
      '402\u00A0KB',
      '700\u00A0MB',
      '1.50\u00A0GB',
      '-18.2\u00A0dBFS',
    ]) {
      expect(text, contains(kept));
    }
    final split = RegExp(r'\d (ms|s|fps|tok|KB|MB|GB|dBFS)\b');
    expect(shown.where(split.hasMatch), isEmpty);
    // Otherwise the facts are the list's, word for word.
    expect(
      shown.map((l) => l.replaceAll('\u00A0', ' ')),
      debugOverlayLines(snapshot.value).map((l) => l.replaceAll('\u00A0', ' ')),
    );
  });
}
