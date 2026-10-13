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

// DebugOverlayHost and DebugOverlayToggle on their own: the switch the host
// hands down, the toggle outside a host, and the app under the panel kept
// as it is while the panel comes and goes. Layout and pointers are in
// debug_overlay_layout_test.dart.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';

const _bump = ValueKey('bump');

/// A screen with state of its own: a count the tests raise.
class _Counter extends StatefulWidget {
  const _Counter();

  @override
  State<_Counter> createState() => _CounterState();
}

class _CounterState extends State<_Counter> {
  int _count = 0;

  @override
  Widget build(BuildContext context) => Align(
    alignment: Alignment.bottomLeft,
    child: TextButton(
      key: _bump,
      onPressed: () => setState(() => _count++),
      child: Text('count $_count'),
    ),
  );
}

/// What a widget under the host reads from it.
class _VisibilityProbe extends StatelessWidget {
  const _VisibilityProbe();

  @override
  Widget build(BuildContext context) =>
      Text('visible: ${DebugOverlayHost.visibilityOf(context)?.value}');
}

Widget _app(ValueNotifier<DiagnosticsSnapshot> snapshot) => MaterialApp(
  builder: (context, child) =>
      DebugOverlayHost(snapshot: snapshot, child: child!),
  home: Scaffold(
    appBar: AppBar(actions: const [DebugOverlayToggle()]),
    body: const Column(
      children: [
        _VisibilityProbe(),
        Expanded(child: _Counter()),
      ],
    ),
  ),
);

IconButton _toggle(WidgetTester tester) =>
    tester.widget<IconButton>(find.byKey(DebugOverlayKeys.toggle));

void main() {
  testWidgets('outside a host: no toggle, and no switch to find', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          appBar: AppBar(actions: const [DebugOverlayToggle()]),
          body: const _VisibilityProbe(),
        ),
      ),
    );

    expect(find.byKey(DebugOverlayKeys.toggle), findsNothing);
    expect(find.byType(IconButton), findsNothing);
    expect(find.text('visible: null'), findsOneWidget);
  });

  testWidgets('shown by default in a debug build; the toggle hides and '
      'shows it, its icon and tooltip follow, and the screen under it keeps '
      'its state', (tester) async {
    final snapshot = ValueNotifier(const DiagnosticsSnapshot());
    addTearDown(snapshot.dispose);
    await tester.pumpWidget(_app(snapshot));
    await tester.pumpAndSettle();

    expect(find.byKey(DebugOverlayKeys.panel), findsOneWidget);
    expect(_toggle(tester).tooltip, 'Hide diagnostics');
    expect(find.byIcon(Icons.bug_report), findsOneWidget);
    await tester.tap(find.byKey(_bump));
    await tester.pump();
    expect(find.text('count 1'), findsOneWidget);

    await tester.tap(find.byKey(DebugOverlayKeys.toggle));
    await tester.pump();
    expect(find.byKey(DebugOverlayKeys.panel), findsNothing);
    expect(_toggle(tester).tooltip, 'Show diagnostics');
    expect(find.byIcon(Icons.bug_report_outlined), findsOneWidget);
    expect(find.text('count 1'), findsOneWidget, reason: 'not rebuilt anew');

    await tester.tap(find.byKey(DebugOverlayKeys.toggle));
    await tester.pump();
    expect(find.byKey(DebugOverlayKeys.panel), findsOneWidget);
    expect(_toggle(tester).tooltip, 'Hide diagnostics');
    expect(find.text('count 1'), findsOneWidget);
  });

  testWidgets("visibilityOf is the host's switch: a widget that reads it "
      'follows it, and setting it shows or hides the panel', (tester) async {
    final snapshot = ValueNotifier(const DiagnosticsSnapshot());
    addTearDown(snapshot.dispose);
    await tester.pumpWidget(_app(snapshot));
    await tester.pumpAndSettle();
    expect(find.text('visible: true'), findsOneWidget);

    final visible = DebugOverlayHost.visibilityOf(
      tester.element(find.byType(_VisibilityProbe)),
    )!;
    visible.value = false;
    await tester.pump();
    expect(find.text('visible: false'), findsOneWidget);
    expect(find.byKey(DebugOverlayKeys.panel), findsNothing);
    expect(_toggle(tester).tooltip, 'Show diagnostics');

    visible.value = true;
    await tester.pump();
    expect(find.text('visible: true'), findsOneWidget);
    expect(find.byKey(DebugOverlayKeys.panel), findsOneWidget);
  });
}
