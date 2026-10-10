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
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/utils/result.dart';

import 'fake_audio_repository.dart';
import 'fake_conversation_repository.dart';

/// The fakes hold to the real repositories' contracts where a caller can
/// get them wrong, so the callers' failure paths are tested against the
/// same rules the app meets: a fake looser than its contract would let a
/// broken failure path pass.
void main() {
  group('FakeConversationRepository (EdgeAiConversationRepository.ask)', () {
    late FakeConversationRepository chat;

    setUp(() => chat = FakeConversationRepository());
    tearDown(() => chat.close());

    Future<AssistantEvent> firstEvent(Stream<AssistantEvent> turn) =>
        turn.first;

    Matcher failedWith<T>() =>
        isA<AssistantFailed>().having((e) => e.error, 'error', isA<T>());

    test('ask is lazy: nothing happens until the stream is listened to', () {
      chat.leaveOpen(kVoiceChatProfile);

      final turn = chat.ask('q');

      expect(chat.prompts, isEmpty);
      expect(chat.isGenerating.value, isFalse);
      expect(turn, isA<Stream<AssistantEvent>>());
    });

    test('a chat that is not open answers AssistantFailed', () async {
      expect(
        await firstEvent(chat.ask('q')),
        failedWith<ConversationNotReadyException>(),
      );
      expect(chat.isGenerating.value, isFalse);
    });

    test('a second ask while a reply runs answers AssistantFailed; the '
        'running turn goes on', () async {
      chat.leaveOpen(kVoiceChatProfile);
      final first = <AssistantEvent>[];
      final sub = chat.ask('one').listen(first.add);
      await pumpEventQueue();
      expect(chat.isGenerating.value, isTrue);

      expect(
        await firstEvent(chat.ask('two')),
        failedWith<ConversationNotReadyException>(),
      );

      chat.emit('a');
      await chat.finish();
      await pumpEventQueue();
      expect(first.whereType<AssistantTextDelta>(), hasLength(1));
      expect(first.last, isA<AssistantDone>());
      await sub.cancel();
    });

    test(
      'an image the chat was built without answers AssistantFailed',
      () async {
        chat.capabilities = const ChatCapabilities(
          modelName: 'Text only',
          images: false,
          tools: false,
        );
        await chat.open(kVoiceChatProfile);

        expect(
          await firstEvent(chat.ask('q', image: Uint8List(4))),
          failedWith<ConversationImageUnsupportedException>(),
        );
      },
    );

    test('a listener that cancels ends the turn; the next ask runs', () async {
      chat.leaveOpen(kVoiceChatProfile);
      final sub = chat.ask('one').listen((_) {});
      await pumpEventQueue();
      expect(chat.isGenerating.value, isTrue);

      await sub.cancel();
      expect(chat.isGenerating.value, isFalse);

      final next = chat.ask('two').listen((_) {});
      await pumpEventQueue();
      expect(chat.isGenerating.value, isTrue);
      await chat.finish();
      await next.cancel();
    });

    test(
      'close is idempotent; an ask afterwards answers AssistantFailed',
      () async {
        chat.leaveOpen(kVoiceChatProfile);

        await chat.close();
        await chat.close();

        expect(
          await firstEvent(chat.ask('q')),
          failedWith<ConversationNotReadyException>(),
        );
      },
    );
  });

  group('FakeAudioRepository (DeviceAudioRepository)', () {
    late FakeAudioRepository audio;

    setUp(() => audio = FakeAudioRepository());
    tearDown(() => audio.close());

    Future<Result<CaptureHandle>> start() => audio.startCapture(
      maxLength: const Duration(seconds: 5),
      onLimit: () {},
    );

    test('beginPlayback fails until a prepare succeeded', () async {
      expect(
        (audio.beginPlayback(24000) as Error<PlaybackHandle>).error,
        isA<PlaybackException>(),
      );

      audio.prepareResult = Result.error(Exception('no output'));
      expect(await audio.prepare(), isA<Error<void>>());
      expect(audio.beginPlayback(24000), isA<Error<PlaybackHandle>>());

      audio.prepareResult = const Result.ok(null);
      expect(await audio.prepare(), isA<Ok<void>>());
      expect(audio.beginPlayback(24000), isA<Ok<PlaybackHandle>>());
    });

    test('a capture prepares, like the real one: playback works after '
        'it', () async {
      expect(await start(), isA<Ok<CaptureHandle>>());
      expect(audio.beginPlayback(24000), isA<Ok<PlaybackHandle>>());
    });

    test(
      'a second startCapture while one starts supersedes the first',
      () async {
        audio.startGate = Completer<void>();
        final first = start();
        final second = start();
        audio.startGate!.complete();

        expect(
          (await first as Error<CaptureHandle>).error,
          isA<AudioSupersededException>(),
        );
        expect(await second, isA<Ok<CaptureHandle>>());
      },
    );

    test('a startCapture cancels a capture still open', () async {
      final first = (await start() as Ok<CaptureHandle>).value as FakeCapture;

      expect(await start(), isA<Ok<CaptureHandle>>());

      expect(first.superseded, isTrue);
      expect(first.cancelled, isTrue);
    });

    FakeCapture opened(Result<CaptureHandle> result) =>
        (result as Ok<CaptureHandle>).value as FakeCapture;

    Matcher superseded<T>() => isA<Error<T>>().having(
      (e) => e.error,
      'error',
      isA<AudioSupersededException>(),
    );

    test('stop after a cancel, or after a newer capture replaced it, is '
        'AudioSupersededException, not an utterance', () async {
      final cancelled = opened(await start());
      await cancelled.cancel();
      expect(await cancelled.stop(), superseded<Utterance>());

      final replaced = opened(await start());
      opened(await start());
      expect(await replaced.stop(), superseded<Utterance>());
    });

    test('cancel during a stop is a no-op: the press keeps its '
        'utterance', () async {
      final capture = opened(await start());

      final stopping = capture.stop();
      await capture.cancel();

      expect(capture.cancelled, isFalse);
      expect(await stopping, isA<Ok<Utterance>>());
    });

    test('close cancels the open capture and stops the playback, which '
        'completes its drain', () async {
      final capture = opened(await start());
      final playback = audio.beginPlayback(24000) as Ok<PlaybackHandle>;
      final reply = playback.value as FakePlayback;
      reply.enqueue(Uint8List(4));

      await audio.close();

      expect(capture.cancelled, isTrue);
      expect(reply.stopped, isTrue);
      await reply.drained.timeout(const Duration(seconds: 1));
    });

    test('a new playback stops the one still playing (barge-in relies on '
        'it); one that drained is left alone', () async {
      expect(await audio.prepare(), isA<Ok<void>>());
      final first =
          (audio.beginPlayback(24000) as Ok<PlaybackHandle>).value
              as FakePlayback;

      final second =
          (audio.beginPlayback(24000) as Ok<PlaybackHandle>).value
              as FakePlayback;
      expect(first.stopped, isTrue);
      await first.drained.timeout(const Duration(seconds: 1));
      expect(second.stopped, isFalse);

      second.completeDrain();
      final logged = audio.log.length;
      audio.beginPlayback(24000);
      expect(second.stopped, isFalse);
      expect(audio.log.sublist(logged), ['beginPlayback(24000)']);
    });

    test('after close every call returns the closed error; close is '
        'idempotent', () async {
      await audio.close();
      await audio.close();

      Matcher closed<T>() => isA<Error<T>>().having(
        (e) => '${e.error}',
        'error',
        contains('AudioRepository closed'),
      );
      expect(await audio.prepare(), closed<void>());
      expect(await audio.requestMicAccess(), closed<void>());
      expect(await start(), closed<CaptureHandle>());
      expect(audio.beginPlayback(24000), closed<PlaybackHandle>());
    });
  });
}
