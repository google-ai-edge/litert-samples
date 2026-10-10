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

import '../../../config/model_catalog.dart';
import '../../../domain/models/assistant_event.dart';
import '../../../domain/models/audio_devices.dart';
import '../../../domain/models/diagnostics_snapshot.dart';
import '../../../domain/models/knowledge.dart';
import '../../../domain/models/live_state.dart';
import '../../../domain/models/llm_image.dart';
import '../../../domain/models/model_id.dart';
import '../../../domain/models/model_state.dart';
import '../../../domain/models/route_decision.dart';
import '../../../domain/models/scene_snapshot.dart';
import '../../../domain/models/skill_step.dart';
import '../../../domain/models/voice.dart';

/// The panel's text, one line per fact. [now] is for tests (the paused
/// timer).
List<String> debugOverlayLines(DiagnosticsSnapshot s, {DateTime? now}) {
  final lines = <String>[];
  for (final id in ModelId.values) {
    // The embedder heads the knowledge-base block below.
    if (id == ModelId.embeddingGemma) continue;
    lines.add(_modelLine(id, s.models[id], active: s.activeStt?.id == id));
  }
  lines.addAll(_liveLines(s, now ?? DateTime.now()));
  lines.add(_generationLine(s.lastGeneration));
  if (s.lastGeneration?.stopLatency case final Duration latency) {
    lines.add('stop ${_ms(latency)}');
  }
  lines.addAll(_imageLines(s));
  lines.addAll(_voiceLines(s));
  lines.addAll(_skillLines(s));
  lines.addAll(_knowledgeLines(s));
  lines.addAll(_cameraLines(s));
  lines.add(_rssLine(s));
  return lines;
}

/// The collapsed panel's text (phones): what must stay on screen so a backend
/// change is never unseen — the chat model with its backend and load time, the
/// detector's backend, fps and latency while it runs (or its state), the last
/// generation's TTFT and tokens per second, and RSS — then how many more lines
/// the full list has (`+23 more`, the handle's label). Built from the same
/// helpers as [debugOverlayLines]. [now] is for tests (the paused timer).
List<String> debugOverlaySummaryLines(DiagnosticsSnapshot s, {DateTime? now}) {
  final at = now ?? DateTime.now();
  final facts = [
    _modelLine(ModelId.chat, s.models[ModelId.chat]),
    ...switch (s.liveState) {
      LiveRunning() => [
        'det ${_detectorLabel(s)} · ${_detectorFps(s.liveStats)} · '
            '${_detectorLatency(s.liveStats)}',
      ],
      _ => _liveLines(s, at),
    },
    _generationLine(s.lastGeneration),
    _rssLine(s),
  ];
  final more = debugOverlayLines(s, now: at).length - facts.length;
  return [...facts, more > 0 ? '+$more more' : 'more'];
}

/// `TTFT 350 ms · 21.5 tok/s (chunks) · chunks 42`, or dashes before the
/// first generation.
String _generationLine(GenerationMetrics? g) {
  if (g == null) return 'TTFT – · tok/s –';
  final rate = g.tokensPerSecond?.toStringAsFixed(1) ?? '–';
  final source = switch (g.tokensPerSecondSource) {
    TokenRateSource.native => 'native',
    TokenRateSource.chunks => 'chunks',
  };
  final ttft = switch (g.timeToFirstToken) {
    null => '–',
    final Duration d => _ms(d),
  };
  return 'TTFT $ttft · $rate tok/s ($source) · chunks ${g.chunks}';
}

String _rssLine(DiagnosticsSnapshot s) =>
    'RSS ${_bytes(s.rssBytes)} · peak ${_bytes(s.peakRssBytes)}';

/// Demo 1's picture and the chat's context: the attachment's size, what the
/// last turn did with its image (sent, re-sent and why, or already in the
/// context) with its estimated tokens, and the context use after that turn.
List<String> _imageLines(DiagnosticsSnapshot s) {
  final lines = <String>[];
  if (s.attachment case final LlmImage a?) {
    lines.add(
      'img ${a.width}×${a.height} PNG ${_size(a.png.length)} · from '
      '${a.sourceWidth}×${a.sourceHeight} ${_size(a.sourceBytes)} · '
      'norm ${_ms(a.normalizeTime)}',
    );
  }
  final g = s.lastGeneration;
  if (g == null) return lines;
  if (g.imageAttached) {
    final what = switch ((g.imageSent, g.imageResent)) {
      (false, _) => 'in context (not re-sent)',
      (true, final ImageLoss why?) => 'resent (lost: ${why.name})',
      (true, null) => 'sent',
    };
    final tokens = g.imageTokens == null ? '' : ' · ≈${g.imageTokens} tok';
    lines.add('img $what$tokens');
  }
  if (g.contextTokens != null || g.contextReset) {
    lines.add(
      'ctx ${g.contextTokens ?? '–'}/'
      '${s.models[ModelId.chat]?.chat?.contextTokens ?? kLlmConfig.maxTokens} '
      'tok · prefill '
      '${g.prefillTokens ?? '–'}'
      '${g.contextReset ? ' · reset (${_resetLabel(g.contextResetReason)})' : ''}',
    );
  }
  return lines;
}

String _resetLabel(ContextResetReason reason) => switch (reason) {
  ContextResetReason.budget => 'budget',
  ContextResetReason.interruptedSkill => 'interrupted skill',
};

String _modelLine(ModelId id, LoadedModelInfo? info, {bool active = false}) =>
    info == null
    ? '${id.spec.displayName}: not loaded'
    : '${info.chat?.name ?? info.modelId} · '
          '${info.detail ?? info.backend.toUpperCase()} · '
          'load ${_ms(info.loadTime)} · warm ${_ms(info.warmUpTime)}'
          '${active ? ' · ACTIVE' : ''}';

/// How a ready index came to be on this launch.
String _kbHow(bool reused, KnowledgeOrigin origin) =>
    switch ((reused, origin)) {
      (true, KnowledgeOrigin.prebuilt) => 'reused (prebuilt)',
      (true, KnowledgeOrigin.device) => 'reused',
      (false, KnowledgeOrigin.prebuilt) => 'prebuilt index installed',
      (false, KnowledgeOrigin.device) => 'indexed',
    };

/// ` · prebuilt index not used: <why>`, or nothing.
String _prebuiltSkipped(String? reason) =>
    reason == null ? '' : ' · prebuilt index not used: $reason';

/// The embedder, the knowledge base's state (index progress) and the
/// newest retrieval's time, top similarity and outcome ("below gate").
/// None when no knowledge base is wired.
List<String> _knowledgeLines(DiagnosticsSnapshot s) {
  final status = s.knowledge;
  if (status == null) return const [];
  final lines = [
    _modelLine(ModelId.embeddingGemma, s.models[ModelId.embeddingGemma]),
    switch (status) {
      KnowledgeWaiting() => 'KB waiting for the embedder',
      KnowledgeUnavailable(:final reason) => 'KB unavailable: $reason',
      KnowledgeIndexing(
        :final done,
        :final total,
        :final percent,
        :final prebuiltSkipped,
      ) =>
        'KB indexing $percent% ($done/$total chunks)'
            '${_prebuiltSkipped(prebuiltSkipped)}',
      KnowledgeReady(
        :final chunks,
        :final reused,
        :final elapsed,
        :final origin,
        :final prebuiltSkipped,
      ) =>
        'KB ready · $chunks chunks · ${_kbHow(reused, origin)} in '
            '${_ms(elapsed)}${_prebuiltSkipped(prebuiltSkipped)}',
      KnowledgeFailed(:final message) => 'KB failed: $message',
    },
  ];
  final r = s.lastRetrieval;
  if (r != null) {
    final time = r.latency == null ? '–' : _ms(r.latency!);
    final top = r.topSimilarity?.toStringAsFixed(2) ?? '–';
    final gate = r.gate?.toStringAsFixed(2) ?? '–';
    lines.add(switch (r.outcome) {
      RetrievalOutcome.used =>
        'RAG $time · top $top · ${r.passages.length} excerpts (gate $gate)',
      RetrievalOutcome.belowGate => 'RAG $time · top $top · below gate $gate',
      RetrievalOutcome.unavailable => 'RAG unavailable: ${r.detail ?? '–'}',
      RetrievalOutcome.failed => 'RAG failed: ${r.detail ?? '–'}',
      RetrievalOutcome.skipped =>
        'RAG skipped: skill question ${r.detail ?? ''}',
    });
  }
  return lines;
}

/// The skills scan (`skills 4 ok / 1 error`) and the last agent
/// turn's tool timings, each step at its time since the turn started:
/// `tools 2 · loadSkill(current-time) 0.84 s → current_time 1.71 s (+2 ms)
/// → text 2.60 s`. None when no skills are wired.
List<String> _skillLines(DiagnosticsSnapshot s) {
  final catalog = s.skills;
  final lines = <String>[
    if (catalog != null)
      catalog.storeError != null
          ? 'skills unavailable: ${catalog.storeError}'
          : 'skills ${catalog.skills.length} ok / ${catalog.errors.length} '
                'error${catalog.errors.length == 1 ? '' : 's'}',
  ];
  final g = s.lastGeneration;
  if (g == null || g.toolRounds == 0) return lines;
  String secs(Duration d) =>
      '${(d.inMilliseconds / 1000).toStringAsFixed(2)} s';
  final parts = <String>[];
  for (final step in g.skillSteps) {
    switch (step) {
      case SkillLoaded(:final name, :final found, :final at):
        parts.add('loadSkill($name${found ? '' : '?'}) ${secs(at)}');
      case IntentCalled(:final intent, :final at):
        parts.add('$intent ${secs(at)}');
      case IntentSucceeded(:final elapsed) when parts.isNotEmpty:
        // The app's own work, on the call it answers.
        parts.last = '${parts.last} (+${elapsed.inMilliseconds} ms)';
      case IntentSucceeded():
        break;
      case IntentFailed(:final intent, :final at):
        parts.add('${intent ?? 'tool'} failed ${secs(at)}');
    }
  }
  if (g.timeToFirstToken case final Duration first) {
    parts.add('text ${secs(first)}');
  }
  lines.add('tools ${g.toolRounds} · ${parts.join(' → ')}');
  return lines;
}

/// `audio in <mic> · out <speaker>`, or what is wrong with either.
String _audioLine(AudioDeviceStatus a) {
  String side(DeviceCheck c) => switch (c) {
    DeviceUnchecked() => 'unchecked',
    DeviceReady(:final name, :final caution) => caution ? '$name (!)' : name,
    DeviceUnavailable(:final message) => 'UNAVAILABLE: $message',
  };
  return 'audio in ${side(a.input)} · out ${side(a.output)}';
}

/// The voice loop: phase, STT time, time to first audio, TTS time per
/// clause, and the last barge-in's silence / drain times.
List<String> _voiceLines(DiagnosticsSnapshot s) {
  String opt(Duration? d) => d == null ? '–' : _ms(d);
  final t = s.lastVoiceTurn;
  final stt = s.activeStt;
  final lines = <String>[
    if (stt != null)
      'STT active: ${stt.modelId}'
          '${stt.switchTime == null ? '' : ' · switch ${_ms(stt.switchTime!)}'}'
    else if (s.sttSwitchError == null)
      'STT active: none (switching…)',
    if (s.sttSwitchError case final String error) 'STT switch failed: $error',
    t == null
        ? 'voice ${s.voicePhase.name} · STT – · first audio –'
        : 'voice ${s.voicePhase.name} · '
              '${t.typed ? 'typed' : 'STT ${opt(t.stt)}'} · '
              'first audio ${opt(t.firstAudio)}'
              '${t.outcome == null ? '' : ' · ${t.outcome!.name}'}',
    _audioLine(s.audioDevices),
  ];
  if (t != null && t.peakDbfs != null) {
    lines.add(
      'gate peak ${t.peakDbfs!.toStringAsFixed(1)} dBFS · voiced '
      '${t.voiced?.inMilliseconds ?? '–'} ms over '
      '${t.gateDbfs?.toStringAsFixed(1) ?? '–'} dBFS',
    );
  }
  if (t != null && t.ttsClauses.isNotEmpty) {
    final clauses = t.ttsClauses.map((d) => d.inMilliseconds).join('/');
    lines.add('TTS/clause $clauses ms (${t.ttsClauses.length})');
  }
  if (s.lastBargeIn case final BargeInMetrics b?) {
    lines.add(
      'barge-in silence ${opt(b.silenced)} · drain ${opt(b.interruptDone)}',
    );
  }
  return lines;
}

/// ` 1280×720 @ 29.7 fps`: the frames the source delivers, before the
/// gate; empty before the first frame.
String _sourceRate(LiveStats stats) => stats.sourceWidth == 0
    ? ''
    : ' ${stats.sourceWidth}×${stats.sourceHeight} @ '
          '${stats.sourceFps.toStringAsFixed(1)} fps';

/// The detector's backend label (`GPU fp32 full`), or `detector`.
String _detectorLabel(DiagnosticsSnapshot s) =>
    s.models[ModelId.yolo26n]?.detail ?? 'detector';

/// `29.7 fps`: frames the detector returned per second.
String _detectorFps(LiveStats stats) => '${stats.fps.toStringAsFixed(1)} fps';

/// `lat 6.0 ms`: send → result on the main isolate.
String _detectorLatency(LiveStats stats) => 'lat ${_f1(stats.latencyMs)} ms';

String _f1(double? v) => v?.toStringAsFixed(1) ?? '–';

/// Demo 3's detector lines; none while live detection is stopped.
List<String> _liveLines(DiagnosticsSnapshot s, DateTime now) {
  final label = _detectorLabel(s);
  final stats = s.liveStats;
  return switch (s.liveState) {
    LiveStopped() => const [],
    LiveStarting() => ['det $label · starting…'],
    LiveRunning(:final source) => [
      'det $label · ${_detectorFps(stats)} · p50 pre/run/post '
          '${_f1(stats.preMs)}/${_f1(stats.runMs)}/${_f1(stats.postMs)} ms',
      'det src $source${_sourceRate(stats)} · ${_detectorLatency(stats)} '
          '· drop busy ${stats.droppedBusy} rate ${stats.droppedRate}',
    ],
    LivePaused(:final reason, :final since) => [
      'det paused ($reason) '
          '${(now.difference(since).inMilliseconds / 1000).toStringAsFixed(1)} s',
    ],
    LiveFailed(:final message) => ['det failed: $message'],
  };
}

/// Demo 3's last question: route and rule, router and snapshot times, and
/// the LLM turns counted since launch.
List<String> _cameraLines(DiagnosticsSnapshot s) {
  final c = s.lastCameraTurn;
  if (c == null) return const [];
  final kind = switch (c.route) {
    FastRoute(:final intent) => 'fast/${intent.name}',
    DetailedRoute() => 'detailed',
  };
  final snap = c.snapshotLatency == null ? '–' : _ms(c.snapshotLatency!);
  final lines = [
    'camq $kind · rule ${c.route.rule} · route '
        '${(c.routeTime.inMicroseconds / 1000).toStringAsFixed(2)} ms · '
        'snap $snap · llm turns ${s.llmTurns}',
  ];
  if (c.image case final EncodedSnapshot image?) {
    lines.add(
      'camq frame #${image.frameId} ${image.width}×${image.height} PNG '
      '${_size(image.png.length)} · enc ${_ms(image.encodeTime)}'
      '${image.unmirrored ? ' · unmirrored' : ''} · TTFT '
      '${c.timeToFirstToken == null ? '–' : _ms(c.timeToFirstToken!)}',
    );
  }
  if (s.lastCameraReset case final Duration reset) {
    lines.add(
      'camq reset ${_ms(reset)}'
      '${s.cameraResetError == null ? '' : ' FAILED: ${s.cameraResetError}'}',
    );
  }
  if (s.frozenFrameId case final int frame) {
    lines.add(
      'view frozen on #$frame · det '
      '${s.liveState is LivePaused ? 'paused' : 'live'}',
    );
  }
  if (c.snapshotError case final String error) {
    lines.add('camq no frame: $error');
  }
  if (c.chatError case final String error) {
    lines.add('camq chat: $error');
  }
  return lines;
}

String _ms(Duration d) => '${d.inMilliseconds} ms';

String _size(int bytes) => bytes >= 1 << 20
    ? '${(bytes / (1 << 20)).toStringAsFixed(2)} MB'
    : '${(bytes / 1024).toStringAsFixed(0)} KB';

String _bytes(int? bytes) => switch (bytes) {
  null => '–',
  >= 1 << 30 => '${(bytes / (1 << 30)).toStringAsFixed(2)} GB',
  _ => '${(bytes / (1 << 20)).toStringAsFixed(0)} MB',
};
