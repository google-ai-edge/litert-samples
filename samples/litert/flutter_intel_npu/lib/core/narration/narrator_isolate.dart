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

// The narrator's worker isolate: owns its own engines (raw LiteRT-LM FFI
// clients, one per backend) and runs every generate() there, so whatever the
// native path blocks on can only block THIS isolate — never the UI isolate
// that owns the frame rate.
//
// Wire protocol is plain maps so nothing here needs Flutter.

import 'dart:async';
import 'dart:isolate';

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../engines/engine_pool.dart';
import '../engines/engine_slot.dart';

/// UI → worker
class NarratorCmd {
  static const load = 'load'; // {backend, modelPath, maxTokens}
  static const open = 'open'; // {backend, maxOutputTokens} → new conversation on that engine
  static const generate = 'generate'; // {prompt, turnId}
  static const close = 'close'; // close the conversation
  static const dispose = 'dispose';
}

/// worker → UI
class NarratorMsg {
  static const ready = 'ready'; // {port}
  static const loaded = 'loaded'; // {backend, loadMs}
  static const opened = 'opened'; // {backend}
  static const chunk = 'chunk'; // {turnId, text}
  static const done = 'done'; // {turnId, ttftMs, tps, inputTokens, outputTokens}
  static const error = 'error'; // {op, message}
}

/// The engine host: everything after the wire. Used by the worker isolate
/// ([narratorMain]) and, with `ACT3_NARRATOR_INLINE=true`, directly on the
/// UI isolate (a diagnostic/fallback mode — see CLAUDE.md, Windows GPU).
class NarratorEngineHost {
  final EnginePool _pool = EnginePool(policy: EnginePolicy.resident, modelPath: '');
  EngineTurn? _turn;

  /// Handles one command; replies (possibly several) go through [send].
  /// Returns false after `dispose`.
  Future<bool> handle(Map<String, Object?> msg, void Function(Map<String, Object?>) send) async {
    final type = msg['type'] as String;
    try {
      switch (type) {
        case NarratorCmd.load:
          final b = PreferredBackend.values.byName(msg['backend'] as String);
          final slot = await _pool.acquire(b, maxTokens: msg['maxTokens'] as int, modelPath: msg['modelPath'] as String);
          send({'type': NarratorMsg.loaded, 'backend': b.name, 'loadMs': slot.loadTime?.inMilliseconds});
        case NarratorCmd.open:
          final b = PreferredBackend.values.byName(msg['backend'] as String);
          final slot = await _pool.acquire(b, maxTokens: msg['maxTokens'] as int, modelPath: msg['modelPath'] as String);
          await _turn?.close();
          _turn = await slot.openTurn(maxOutputTokens: msg['maxOutputTokens'] as int, topK: 1);
          send({'type': NarratorMsg.opened, 'backend': b.name});
        case NarratorCmd.generate:
          final t = _turn;
          if (t == null) throw StateError('no open conversation');
          final id = msg['turnId'] as int;
          final sw = Stopwatch()..start();
          int? ttft;
          await for (final chunk in t.generate(msg['prompt'] as String)) {
            ttft ??= sw.elapsedMilliseconds;
            send({'type': NarratorMsg.chunk, 'turnId': id, 'text': chunk});
          }
          final m = t.metrics;
          send({
            'type': NarratorMsg.done,
            'turnId': id,
            'ttftMs': ttft ?? sw.elapsedMilliseconds,
            'wallMs': sw.elapsedMilliseconds,
            'tps': m.tokensPerSecond,
            'inputTokens': m.inputTokens,
            'outputTokens': m.outputTokens,
          });
        case NarratorCmd.close:
          await _turn?.close();
          _turn = null;
        case NarratorCmd.dispose:
          await _turn?.close();
          await _pool.releaseAll();
          return false;
      }
    } catch (e, st) {
      send({'type': NarratorMsg.error, 'op': type, 'turnId': msg['turnId'], 'message': '$e\n$st'});
    }
    return true;
  }
}

Future<void> narratorMain(SendPort toUi) async {
  final inbox = ReceivePort();
  toUi.send({'type': NarratorMsg.ready, 'port': inbox.sendPort});
  final host = NarratorEngineHost();
  await for (final raw in inbox) {
    final alive = await host.handle(raw as Map<String, Object?>, toUi.send);
    if (!alive) {
      inbox.close();
      return;
    }
  }
}
