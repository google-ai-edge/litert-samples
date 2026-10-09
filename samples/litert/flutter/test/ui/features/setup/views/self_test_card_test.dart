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

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/self_test.dart';
import 'package:litert_edge_demos/selftest/self_test_in_app.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/self_test_view_model.dart';
import 'package:litert_edge_demos/ui/features/setup/views/self_test_card.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_self_test_launcher.dart';

void main() {
  testWidgets('Run self-test shows the report, where it was saved, and Copy', (
    tester,
  ) async {
    final launcher = FakeSelfTestLauncher(
      const Result.ok(
        SelfTestOutcome(
          text: '===== SELFTEST BEGIN =====\nresult     FAIL · exit 1\n',
          passed: false,
          reportPath: '/data/selftest/selftest-x.txt',
        ),
      ),
    );
    final vm = SelfTestViewModel(launcher: launcher);
    addTearDown(vm.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ListView(children: [SelfTestCard(viewModel: vm)]),
        ),
      ),
    );
    expect(find.byKey(SelfTestKeys.copy), findsNothing);

    await tester.tap(find.byKey(SelfTestKeys.run));
    await tester.pump();
    await tester.pump();

    expect(launcher.runs, 1);
    expect(tester.widget<Text>(find.byKey(SelfTestKeys.result)).data, 'FAIL');
    expect(find.text('Saved to /data/selftest/selftest-x.txt'), findsOneWidget);
    expect(
      tester.widget<SelectableText>(find.byKey(SelfTestKeys.report)).data,
      contains('SELFTEST BEGIN'),
    );
    expect(find.byKey(SelfTestKeys.copy), findsOneWidget);
  });

  testWidgets('a run that did not end: its report stays, Run is disabled and '
      'says to restart the app', (tester) async {
    final launcher = FakeSelfTestLauncher(
      const Result.ok(
        SelfTestOutcome(
          text: '$kSelfTestStillRunning\n===== SELFTEST BEGIN =====\n',
          passed: false,
          stillRunning: true,
        ),
      ),
    )..stuckAfterRun = true;
    final vm = SelfTestViewModel(launcher: launcher);
    addTearDown(vm.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ListView(children: [SelfTestCard(viewModel: vm)]),
        ),
      ),
    );

    await tester.tap(find.byKey(SelfTestKeys.run));
    await tester.pump();
    await tester.pump();

    final run = tester.widget<FilledButton>(find.byKey(SelfTestKeys.run));
    expect(run.onPressed, isNull);
    expect(find.textContaining('restart the app'), findsWidgets);
    expect(
      vm.blockedReason,
      contains('restart the app before running it again'),
    );
    expect(
      tester.widget<SelectableText>(find.byKey(SelfTestKeys.report)).data,
      contains('SELFTEST BEGIN'),
      reason: 'the stuck report is kept, not replaced by an error',
    );
    expect(launcher.runs, 1);
  });
}
