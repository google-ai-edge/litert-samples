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
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/message_bubble.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/skill_steps.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/skills_sheet.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../../integration_test/support/skill_fixtures.dart';
import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_image_input.dart';
import '../../../../fakes/fake_skill_store.dart';
import '../../../../fakes/fake_speech.dart';

SkillCatalog catalogOf(
  List<String> texts, {
  List<SkillLoadError> errors = const [],
}) => SkillCatalog(
  directory: '/docs/skills',
  skills: [
    for (final text in texts)
      LoadedSkill(
        skill: parseSkillMd(text),
        path: '${parseSkillMd(text).name}/SKILL.md',
      ),
  ],
  errors: errors,
  fingerprint: '${texts.length}/${errors.length}',
);

void main() {
  testWidgets('a reply with skill steps shows them, with the arguments and '
      'the result', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: MessageBubble(
            entry: ChatEntry(
              role: ChatRole.assistant,
              text: 'It is just after two.',
              steps: [
                SkillLoaded(
                  'current-time',
                  found: true,
                  at: Duration(milliseconds: 840),
                ),
                IntentCalled(
                  'current_time',
                  '{}',
                  at: Duration(milliseconds: 1710),
                ),
                IntentSucceeded(
                  'current_time',
                  'It is 2:03 PM.',
                  elapsed: Duration(milliseconds: 2),
                  at: Duration(milliseconds: 1712),
                ),
              ],
            ),
          ),
        ),
      ),
    );

    expect(find.byKey(SkillStepKeys.panel), findsOneWidget);
    expect(find.text('loadSkill(current-time) · 0.8 s'), findsOneWidget);
    expect(find.text('runIntent(current_time, {}) · 1.7 s'), findsOneWidget);
    expect(find.text('It is 2:03 PM. (2 ms)'), findsOneWidget);
    expect(find.text('It is just after two.'), findsOneWidget);
  });

  testWidgets('the streaming bubble shows the running turn\'s steps before '
      'any text, so silent tool rounds do not look like a hang', (
    tester,
  ) async {
    final text = ValueNotifier('');
    final steps = ValueNotifier<List<SkillStep>>(const []);
    addTearDown(text.dispose);
    addTearDown(steps.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StreamingBubble(text: text, steps: steps),
        ),
      ),
    );
    expect(find.byKey(SkillStepKeys.panel), findsNothing);
    expect(find.text('…'), findsOneWidget);

    steps.value = const [
      SkillLoaded('current-time', found: true, at: Duration(milliseconds: 900)),
    ];
    await tester.pump();

    expect(find.byKey(SkillStepKeys.panel), findsOneWidget);
    expect(find.text('loadSkill(current-time) · 0.9 s'), findsOneWidget);
    expect(find.text('…'), findsOneWidget);
  });

  testWidgets('the Skills sheet lists skills and broken files; Reload picks '
      'up a new skill and says the conversation started over', (tester) async {
    final timeText =
        '---\nname: current-time\ndescription: The time.\n---\nCall the '
        '`run_intent` tool with intent `current_time`.';
    final store = FakeSkillStore(
      catalog: catalogOf(
        [timeText],
        errors: const [
          SkillLoadError(
            path: 'broken/SKILL.md',
            message: "Invalid format: expected a '---' fenced YAML frontmatter block.",
          ),
        ],
      ),
    );
    final skills = SkillRepository(store: store);
    final conversation = FakeConversationRepository();
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    final speech = await loadedSpeech(
      recognizer: FakeRecognizer(),
      synthesizer: RecordingSynth(),
    );
    final vm = VoiceChatViewModel(
      activateStt: (_) async => const Result.ok(null),
      conversation: conversation,
      diagnostics: diagnostics,
      images: fakeImageRepository(FakeImageInputService()),
      skills: skills,
      assistant: VoiceAssistant(
        speech: speech,
        audio: FakeAudioRepository()..autoDrain = true,
        responders: ChatTurnResponder(conversation: conversation),
        diagnostics: diagnostics,
      ),
    );
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: SkillsSheet(viewModel: vm)),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.byKey(SkillsSheetKeys.skill('current-time')), findsOneWidget);
    expect(
      find.byKey(SkillsSheetKeys.error('broken/SKILL.md')),
      findsOneWidget,
    );
    expect(find.textContaining("expected a '---' fenced"), findsOneWidget);
    expect(find.text('/docs/skills'), findsOneWidget);

    store.catalog = catalogOf([timeText, kidClockSkillMd]);
    await tester.tap(find.byKey(SkillsSheetKeys.reload));
    await tester.pumpAndSettle();

    expect(find.byKey(SkillsSheetKeys.skill('kid-clock')), findsOneWidget);
    expect(find.byKey(SkillsSheetKeys.error('broken/SKILL.md')), findsNothing);
    expect(
      tester.widget<Text>(find.byKey(SkillsSheetKeys.status)).data,
      contains('started over'),
    );
    expect(conversation.openedSkills.last.map((s) => s.name), [
      'current-time',
      'kid-clock',
    ]);

    vm.dispose();
    await tester.pumpAndSettle();
    diagnostics.dispose();
    models.dispose();
    skills.dispose();
  });
}
