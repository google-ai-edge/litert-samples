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

import 'assistant_event.dart';
import 'knowledge.dart';
import 'skill_step.dart';

/// Demo 1's side channel of a voice turn: facts the reply text does not
/// carry. Skill steps join it.
sealed class const ChatSideEvent();

/// The model's turn ended (normally or stopped) with these timings.
final class const ChatGenerationDone(final GenerationMetrics metrics)
    extends ChatSideEvent;

/// The turn's knowledge-base retrieval, before the model is asked:
/// the excerpts in the prompt (cited as `[n]`) or why there are none.
final class const ChatRetrieval(final Retrieval retrieval)
    extends ChatSideEvent;

/// The conversation started over before this turn ([reason]: the context
/// was full, or an interrupted skill call left the chat unusable): the user
/// must be told, whatever happens to the turn afterwards.
final class const ChatContextReset({
  final ContextResetReason reason = ContextResetReason.budget,
}) extends ChatSideEvent;

/// One skill step of the turn (loadSkill, runIntent, its result), as
/// it happens.
final class const ChatSkillStep(final SkillStep step) extends ChatSideEvent;
