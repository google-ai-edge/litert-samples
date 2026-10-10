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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation/open_queue.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/utils/result.dart';

Skill _skill(String name) => parseSkillMd(
  '---\nname: $name\ndescription: $name for tests\n---\nSay $name.\n',
);

/// One call of the queue's rebuild, held until the test completes it.
final class _Build {
  _Build(this.profile, this.skills);

  final ConversationProfile profile;
  final List<Skill> skills;
  final Completer<Result<void>> result = Completer();
}

Future<void> settle() async {
  for (var i = 0; i < 10; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

void main() {
  late List<_Build> builds;
  late OpenQueue queue;
  final time = _skill('time');
  final device = _skill('device');

  setUp(() {
    builds = [];
    queue = OpenQueue((profile, skills) {
      final build = _Build(profile, skills);
      builds.add(build);
      return build.result.future;
    });
  });

  test('an open builds its profile with its skills and returns the '
      'result', () async {
    final opening = queue.open(kVoiceChatProfile, [time]);
    await settle();

    expect(builds.single.profile, kVoiceChatProfile);
    expect(builds.single.skills, [time]);
    builds.single.result.complete(const Result.ok(null));
    expect(await opening, isA<Ok<void>>());
  });

  test('the skills are copied: a later change to the caller\'s list does '
      'not reach the build', () async {
    final skills = [time];
    unawaited(queue.open(kVoiceChatProfile, skills));
    skills.add(device);
    await settle();

    expect(builds.single.skills, [time]);
    expect(() => builds.single.skills.add(device), throwsUnsupportedError);
    builds.single.result.complete(const Result.ok(null));
  });

  test('one rebuild at a time; of the opens queued meanwhile only the newest '
      'is built, earlier ones for another profile are superseded', () async {
    final first = queue.open(kVoiceChatProfile, const []);
    await settle();
    final second = queue.open(kCameraProfile, const []);
    final third = queue.open(kVoiceChatProfile, [device]);
    await settle();
    expect(builds, hasLength(1), reason: 'the first build still runs');

    builds.first.result.complete(const Result.ok(null));
    await settle();

    expect(builds, hasLength(2));
    expect(builds.last.profile, kVoiceChatProfile);
    expect(builds.last.skills, [device]);
    builds.last.result.complete(const Result.ok(null));
    expect(await first, isA<Ok<void>>());
    expect(
      await second,
      isA<Error<void>>().having(
        (e) => e.error,
        'error',
        isA<ConversationSupersededException>().having(
          (e) => '$e',
          'message',
          'open(camera) was replaced by open(voice-chat) before it ran',
        ),
      ),
    );
    expect(await third, isA<Ok<void>>());
  });

  test('a superseded open for the same profile shares the newest result, '
      'failures included', () async {
    final blocker = queue.open(kCameraProfile, const []);
    await settle();
    final a = queue.open(kVoiceChatProfile, const []);
    final b = queue.open(kVoiceChatProfile, [time]);
    builds.first.result.complete(const Result.ok(null));
    await settle();
    const failure = ConversationNotReadyException('no');
    builds.last.result.complete(const Result.error(failure));

    expect(await blocker, isA<Ok<void>>());
    for (final result in [await a, await b]) {
      expect(
        result,
        isA<Error<void>>().having((e) => e.error, 'error', failure),
      );
    }
    expect(builds, hasLength(2));
  });

  test('a reset rebuilds the requested profile with the newest open\'s '
      'skills', () async {
    unawaited(queue.open(kVoiceChatProfile, [time, device]));
    await settle();
    builds.single.result.complete(const Result.ok(null));
    await settle();

    final resetting = queue.reset(kVoiceChatProfile);
    await settle();

    expect(builds, hasLength(2));
    expect(builds.last.profile, kVoiceChatProfile);
    expect(builds.last.skills, [time, device]);
    builds.last.result.complete(const Result.ok(null));
    expect(await resetting, isA<Ok<void>>());
  });

  test('a reset for a profile that is no longer the requested one is moot: '
      'Ok, nothing built', () async {
    unawaited(queue.open(kVoiceChatProfile, const []));
    await settle();
    final late = queue.reset(kVoiceChatProfile);
    final switching = queue.open(kCameraProfile, const []);
    builds.single.result.complete(const Result.ok(null));
    await settle();

    expect(await late, isA<Ok<void>>());
    expect(builds, hasLength(2));
    expect(builds.last.profile, kCameraProfile);
    builds.last.result.complete(const Result.ok(null));
    expect(await switching, isA<Ok<void>>());
  });

  test('after forgetRequested every reset is moot until the next '
      'open', () async {
    unawaited(queue.open(kVoiceChatProfile, const []));
    await settle();
    builds.single.result.complete(const Result.ok(null));
    await settle();

    queue.forgetRequested();

    expect(await queue.reset(kVoiceChatProfile), isA<Ok<void>>());
    expect(builds, hasLength(1));
  });

  test('a rebuild that throws fails its requests, is logged, and the queue '
      'goes on', () async {
    final log = <String>[];
    final originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) => log.add(message ?? '');
    addTearDown(() => debugPrint = originalDebugPrint);
    final throwing = OpenQueue((_, _) async => throw StateError('native'));

    final failed = await throwing.open(kVoiceChatProfile, const []);

    expect(
      failed,
      isA<Error<void>>().having(
        (e) => '${e.error}',
        'error',
        'Bad state: native',
      ),
    );
    final again = await throwing.open(kCameraProfile, const []);
    expect(again, isA<Error<void>>(), reason: 'served again, not stuck');
    expect(
      log.where((l) => l.startsWith('[Conversation] rebuild failed: ')),
      hasLength(2),
    );
  });

  test('drain completes at once when idle, otherwise after every queued '
      'request, ones queued meanwhile included', () async {
    await queue.drain();

    unawaited(queue.open(kVoiceChatProfile, const []));
    await settle();
    var drained = false;
    unawaited(queue.drain().then((_) => drained = true));
    unawaited(queue.open(kCameraProfile, const []));
    builds.first.result.complete(const Result.ok(null));
    await settle();
    expect(drained, isFalse, reason: 'the second open is being built');

    builds.last.result.complete(const Result.ok(null));
    await settle();
    expect(drained, isTrue);
  });

  test('inside refusing, an open fails at once without becoming the '
      'requested one and a reset is moot; what was queued before still '
      'builds; nested bodies refuse until the last ends', () async {
    final queued = queue.open(kVoiceChatProfile, [time]);
    final outer = Completer<void>();
    final inner = Completer<void>();
    final refusing = queue.refusing('changing', () => outer.future);
    final nested = queue.refusing('changing', () => inner.future);

    final refused = await queue.open(kCameraProfile, const []);
    expect(
      refused,
      isA<Error<void>>().having((e) => '${e.error}', 'error', 'changing'),
    );
    expect(await queue.reset(kVoiceChatProfile), isA<Ok<void>>());
    await settle();
    expect(builds.single.profile, kVoiceChatProfile, reason: 'queued before');
    builds.single.result.complete(const Result.ok(null));
    expect(await queued, isA<Ok<void>>());

    inner.complete();
    await nested;
    expect(await queue.open(kCameraProfile, const []), isA<Error<void>>());

    outer.complete();
    await refusing;
    // Taken again; the refused open did not replace the requested profile:
    // a reset for the voice profile still rebuilds it with its skills.
    final reset = queue.reset(kVoiceChatProfile);
    await settle();
    expect(builds.last.profile, kVoiceChatProfile);
    expect(builds.last.skills, [time]);
    builds.last.result.complete(const Result.ok(null));
    expect(await reset, isA<Ok<void>>());
  });

  test('a refusing body that ends gives the reason back to the one still '
      'running, nested or overlapping', () async {
    Future<String> refusedWith() async =>
        switch (await queue.open(kCameraProfile, const [])) {
          Error(:final error) => '$error',
          Ok() => 'ok',
        };

    final closing = Completer<void>();
    final changing = Completer<void>();
    final outer = queue.refusing('closed', () => closing.future);
    final inner = queue.refusing('changing', () => changing.future);
    expect(await refusedWith(), 'changing', reason: 'the newest reason');

    changing.complete();
    await inner;
    expect(await refusedWith(), 'closed', reason: 'the outer reason again');

    // Overlapping: the first body ends while the second still runs.
    final releasing = Completer<void>();
    final second = queue.refusing('releasing', () => releasing.future);
    closing.complete();
    await outer;
    expect(await refusedWith(), 'releasing');

    releasing.complete();
    await second;
    final opened = queue.open(kCameraProfile, const []);
    await settle();
    builds.last.result.complete(const Result.ok(null));
    expect(await opened, isA<Ok<void>>());
  });
}
