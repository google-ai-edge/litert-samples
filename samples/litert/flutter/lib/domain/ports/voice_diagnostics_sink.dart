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

// A port: an interface that the domain and the view models depend on and an
// outer layer implements. Every port lives in lib/domain/ports/;
// function-typed dependencies stay next to their one consumer.

import '../models/voice.dart';

/// Where the voice turn machine (`VoiceAssistant`) reports its phases and
/// each turn's figures, for the debug overlay. `DiagnosticsRepository`
/// implements it.
abstract interface class VoiceDiagnosticsSink {
  /// The assistant's phase (a few changes per turn).
  void recordVoicePhase(TurnPhase phase);

  /// The running or finished turn's timings, as they become known.
  void recordVoiceTurn(VoiceTurnMetrics metrics);

  /// A barge-in's cost, as the stop is confirmed and the drain ends.
  void recordBargeIn(BargeInMetrics metrics);
}
