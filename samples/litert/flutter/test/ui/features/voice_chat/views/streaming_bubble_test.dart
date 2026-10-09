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
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/message_bubble.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_skill_store.dart';
import '../../../../fakes/fake_speech.dart';

void main() {
  testWidgets(
    'StreamingBubble shows each token without rebuilding its parent',
    (tester) async {
      final text = ValueNotifier('');
      addTearDown(text.dispose);
      var parentBuilds = 0;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Builder(
              builder: (context) {
                parentBuilds++;
                return StreamingBubble(text: text);
              },
            ),
          ),
        ),
      );
      expect(find.text('…'), findsOneWidget, reason: 'placeholder before text');

      text.value = 'Hel';
      await tester.pump();
      expect(find.text('Hel'), findsOneWidget);

      text.value = 'Hello';
      await tester.pump();
      expect(find.text('Hello'), findsOneWidget);
      expect(parentBuilds, 1);
    },
  );

  testWidgets('VoiceChatScreen streams a typed turn into the bubble and '
      'commits the reply', (tester) async {
    final conversation = FakeConversationRepository();
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    final speech = (await tester.runAsync(
      () => loadedSpeech(
        recognizer: FakeRecognizer(),
        synthesizer: RecordingSynth(),
      ),
    ))!;
    final skills = SkillRepository(store: FakeSkillStore());
    final viewModel = VoiceChatViewModel(
      activateStt: (_) async => const Result.ok(null),
      conversation: conversation,
      diagnostics: diagnostics,
      images: fakeImageRepository(),
      skills: skills,
      assistant: VoiceAssistant(
        speech: speech,
        audio: FakeAudioRepository()..autoDrain = true,
        responders: ChatTurnResponder(conversation: conversation),
        diagnostics: diagnostics,
      ),
    );

    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: viewModel,
        child: const MaterialApp(home: VoiceChatScreen()),
      ),
    );
    await tester.pump(); // open() completes

    await tester.enterText(find.byKey(ChatKeys.input), 'Capital of France?');
    await tester.tap(find.byKey(ChatKeys.send));
    await tester.pump();

    expect(conversation.prompts, ['Capital of France?']);
    expect(find.byKey(ChatKeys.streamingBubble), findsOneWidget);
    expect(find.byKey(ChatKeys.stop), findsOneWidget);
    expect(find.byKey(ChatKeys.send), findsNothing);

    conversation.emit('Par');
    await tester.pump();
    expect(
      find.descendant(
        of: find.byKey(ChatKeys.streamingBubble),
        matching: find.text('Par'),
      ),
      findsOneWidget,
    );

    conversation.emit('is');
    // Real async for the turn's end: VoiceSession awaits the cancel of a
    // finished subscription, which completes on the root zone's microtask
    // queue that testWidgets' fake clock never runs.
    await tester.runAsync(() async {
      await conversation.finish();
      await pumpEventQueue();
    });
    await tester.pump();

    expect(find.byKey(ChatKeys.streamingBubble), findsNothing);
    expect(find.text('Paris'), findsOneWidget);
    expect(find.byKey(ChatKeys.send), findsOneWidget);

    // Tear down in reverse order; flush diagnostics' zero-length timer first.
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pump(Duration.zero);
    viewModel.dispose();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
    await conversation.close();
  });
}
