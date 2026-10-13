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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_knowledge.dart';
import '../../../../fakes/fake_skill_store.dart';
import '../../../../fakes/fake_speech.dart';

Future<void> settle() => pumpEventQueue();

Retrieval used(List<Passage> passages) => Retrieval(
  outcome: RetrievalOutcome.used,
  passages: passages,
  candidates: passages,
  gate: 0.4,
  latency: const Duration(milliseconds: 130),
);

/// A reply's retrieval and citations reach the committed entry, the
/// overlay, and never the speaker.
void main() {
  late FakeConversationRepository conversation;
  late FakeAudioRepository audio;
  late RecordingSynth synth;
  late SpeechRepository speech;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeRetriever retriever;
  late SkillRepository skills;
  late VoiceChatViewModel viewModel;

  setUp(() async {
    conversation = FakeConversationRepository();
    audio = FakeAudioRepository()..autoDrain = true;
    synth = RecordingSynth();
    speech = await loadedSpeech(
      recognizer: FakeRecognizer(),
      synthesizer: synth,
    );
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    skills = SkillRepository(store: FakeSkillStore());
    retriever = FakeRetriever(
      Result.ok(used([passage(1), passage(2), passage(3)])),
    );
    viewModel = VoiceChatViewModel(
      activateStt: (_) async => const Result.ok(null),
      conversation: conversation,
      diagnostics: diagnostics,
      images: fakeImageRepository(),
      skills: skills,
      assistant: VoiceAssistant(
        speech: speech,
        audio: audio,
        responders: ChatTurnResponder(
          conversation: conversation,
          retriever: retriever,
        ),
        diagnostics: diagnostics,
      ),
    );
    await settle();
  });

  tearDown(() async {
    viewModel.dispose();
    await settle();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
  });

  Future<void> typedTurn(String question, String reply) async {
    final sending = viewModel.send.execute(question);
    await settle();
    conversation.emit(reply);
    await conversation.finish();
    await sending;
  }

  test('the reply carries its excerpts and which ones it cited; TTS never '
      'reads a marker', () async {
    await typedTurn(
      'What is LiteRT?',
      'LiteRT runs models on device [1]. It is fast [3].',
    );

    final entry = viewModel.entries.last;
    expect(entry.role, ChatRole.assistant);
    expect(entry.text, 'LiteRT runs models on device [1]. It is fast [3].');
    final knowledge = entry.knowledge!;
    expect(knowledge.retrieval.outcome, RetrievalOutcome.used);
    expect(knowledge.retrieval.passages, hasLength(3));
    expect(knowledge.cited, {1, 3});
    expect(synth.synthesized, isNotEmpty);
    for (final clause in synth.synthesized) {
      expect(clause, isNot(contains('[')), reason: clause);
    }
    expect(diagnostics.latest.lastRetrieval?.outcome, RetrievalOutcome.used);
    expect(viewModel.entries.first.knowledge, isNull, reason: 'user entry');
  });

  test('an unavailable knowledge base is on the reply, not silent', () async {
    retriever.result = const Result.ok(
      Retrieval(
        outcome: RetrievalOutcome.unavailable,
        detail: 'The built-in EmbeddingGemma is not in this app bundle',
      ),
    );

    await typedTurn('What is LiteRT?', 'A runtime.');

    final knowledge = viewModel.entries.last.knowledge!;
    expect(knowledge.retrieval.outcome, RetrievalOutcome.unavailable);
    expect(knowledge.retrieval.detail, contains('not in this app bundle'));
    expect(conversation.prompts.single, 'What is LiteRT?');
  });

  test("a failed turn's retrieval never lands on the next reply", () async {
    final sending = viewModel.send.execute('First?');
    await settle();
    await conversation.fail(Exception('GPU lost'));
    await sending;

    retriever.result = Result.ok(
      Retrieval(
        outcome: RetrievalOutcome.belowGate,
        candidates: [passage(9, similarity: 0.2)],
        gate: 0.4,
        latency: const Duration(milliseconds: 100),
      ),
    );
    await typedTurn('Second?', 'Plain answer.');

    expect(
      viewModel.entries.last.knowledge!.retrieval.outcome,
      RetrievalOutcome.belowGate,
    );
  });

  test('a turn stopped during its retrieval: the late retrieval belongs to '
      'no entry, the next reply has its own', () async {
    retriever.gate = Completer<void>();
    final sending = viewModel.send.execute('First?');
    await settle();
    // The stop drains the turn, which ends once the (uncancellable, ~130 ms)
    // retrieval returns.
    final stopping = viewModel.stop.execute();
    await settle();
    retriever.gate!.complete();
    retriever.gate = null;
    await stopping;
    await sending;
    await settle();
    expect(
      viewModel.entries.where((e) => e.knowledge != null),
      isEmpty,
      reason: 'the stopped turn committed before its retrieval returned',
    );

    retriever.result = Result.ok(used([passage(7)]));
    await typedTurn('Second?', 'Answer [1].');

    final knowledge = viewModel.entries.last.knowledge!;
    expect(knowledge.retrieval.passages.single.id, 'litert-overview.md#7');
    expect(knowledge.cited, {1});
  });
}
