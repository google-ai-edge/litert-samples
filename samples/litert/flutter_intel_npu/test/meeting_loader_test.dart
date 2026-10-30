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

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/features/act4_agent/meeting_loader.dart';

String _ws(String company) => jsonEncode({'company': company});

Matcher _error(String part) => throwsA(isA<FormatException>().having((e) => e.message, 'message', contains(part)));

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late Directory dir;
  late String meeting;
  String file(String name, String text) {
    final f = File('${dir.path}${Platform.pathSeparator}$name')..writeAsStringSync(text);
    return f.path;
  }

  setUp(() {
    dir = Directory.systemTemp.createTempSync('act4_loader');
    meeting = file('weekly.sync.txt', 'Weekly sync\nAna: ship it');
  });
  tearDown(() => dir.deleteSync(recursive: true));

  group('which workspace goes with a meeting', () {
    test('none: the empty workspace, no source', () async {
      final task = await MeetingLoader.fromFile(meeting);
      expect(task.meeting.title, 'Weekly sync');
      expect(task.workspace.isEmpty, isTrue);
      expect(task.workspace.source, isNull);
    });

    test('workspace.json in the folder', () async {
      final shared = file('workspace.json', _ws('Shared'));
      final task = await MeetingLoader.fromFile(meeting);
      expect(task.workspace.company, 'Shared');
      expect(task.workspace.source, shared);
    });

    test('<name>.workspace.json wins over workspace.json', () async {
      file('workspace.json', _ws('Shared'));
      final own = file('weekly.sync.workspace.json', _ws('Own'));
      expect(MeetingLoader.sidecarOf(meeting), own);
      final task = await MeetingLoader.fromFile(meeting);
      expect(task.workspace.company, 'Own');
      expect(task.workspace.source, own);
    });

    test('the configured one (ACT4_WORKSPACE) wins over both', () async {
      file('workspace.json', _ws('Shared'));
      file('weekly.sync.workspace.json', _ws('Own'));
      final configured = file('elsewhere.json', _ws('Configured'));
      final task = await MeetingLoader.fromFile(meeting, workspacePath: configured);
      expect(task.workspace.company, 'Configured');
      expect(task.workspace.source, configured);
    });

    test('another meeting\'s sidecar is not this one\'s', () async {
      file('other.workspace.json', _ws('Other'));
      expect(MeetingLoader.sidecarOf(meeting), isNull);
    });
  });

  group('errors name the file', () {
    test('a workspace that does not parse', () async {
      final bad = file('workspace.json', '{"calender": []}');
      await expectLater(MeetingLoader.fromFile(meeting), _error('$bad: workspace.calender: unknown key'));
    });

    test('a configured workspace that is missing', () async {
      final missing = '${dir.path}${Platform.pathSeparator}missing.json';
      await expectLater(
        MeetingLoader.fromFile(meeting, workspacePath: missing),
        throwsA(isA<FileSystemException>().having((e) => e.path, 'path', missing)),
      );
    });

    test('a transcript that is not text', () async {
      final latin1 = File('${dir.path}${Platform.pathSeparator}old.txt')..writeAsBytesSync([0x5A, 0x6F, 0xEB]);
      await expectLater(MeetingLoader.fromFile(latin1.path), _error('old.txt: not UTF-8 text'));
    });
  });

  group('the bundled example', () {
    test('opens with its own workspace, which has no file path', () async {
      final task = await MeetingLoader.example();
      expect(task.meeting.title, 'Nimbus Notes for Android 3.0: beta readiness');
      expect(task.workspace.internalDomain, 'nimbus.io');
      expect(task.workspace.source, isNull);
    });

    test('a configured workspace replaces the bundled one', () async {
      final configured = file('ws.json', _ws('Configured'));
      final task = await MeetingLoader.example(workspacePath: configured);
      expect(task.workspace.company, 'Configured');
      expect(task.workspace.source, configured);
    });
  });
}
