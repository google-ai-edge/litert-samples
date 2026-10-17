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

// Knowledge-base retrieval on a real target (macOS first): the built-in
// EmbeddingGemma 2 740m on the CPU, loaded as setup loads it (audio
// declared; its .litertlm found as setup finds it), the real sqlite-vec store
// and the real KnowledgeRepository indexing the bundled assets/kb, then the
// golden set (test_assets/kb_golden.json: on-topic and off-topic
// questions, and lines of everyday chat) typed, spoken (by the app's own
// TTS, Inflect-nano-v2, into memory at start-up: support/question_audio.dart,
// the code that makes the shipped clips) and transcribed by the app's Whisper
// with the voice chat's corrections.
//
//   flutter test integration_test/kb_retrieval_test.dart -d macos
//
// For honest latencies keep the test app out of App Nap first:
//   defaults write com.google.ai.edge.examples.litertEdgeDemos NSAppSleepDisabled -bool YES
//
// 1. A fresh index embedded on this device (its own directory, not the
//    app's; the prebuilt index kept out): every chunk embedded and stored,
//    the marker written.
// 2. A second repository over the same files (the next launch): reused, no
//    embedding. The golden set's answers from it are the reference.
// 3. The prebuilt index (assets/kb_index, tool/build_kb_index.sh) installed
//    into an empty directory, as on a first launch: nothing embedded, and the
//    golden set gets the same answers: the same top document for every
//    on-topic question, the same gate decision (top hit at or above
//    kKbMinSimilarityText, or not) for every question, and scores within
//    1e-2 of the device's. The prebuilt vectors come from the build
//    machine's CPU, so on another architecture near-equal chunks can swap
//    places; identical ranked lists are printed as information only.
// 4. Typed (prebuilt index): an on-topic question is answered only when its
//    expected document is among the hits at or above the gate, as the app
//    gates each hit (KnowledgeRepository.retrieve): every one at
//    kKbMinSimilarityText. Off-topic: the best hit below the gate for every
//    one (as lib/config/knowledge_config.dart states).
// 5. The skill trigger phrases of test_assets/skill_trigger_similarity.json
//    against the prebuilt index: their best match, printed as that file's
//    JSON (`KB_SKILL_TRIGGERS {...}`) to paste back after changing the
//    embedder, the gate, the documents or the skills.
// 6. Spoken, embedded as audio (KnowledgeQuery.audio, the clip's WAV as it
//    is): every on-topic question answered at kKbMinSimilarityAudio and
//    every off-topic one below it (as knowledge_config.dart states). The
//    sweep prints the clean range the gate was chosen from.
//    Then the everyday chat, spoken the same way: every line's best hit below
//    kKbMinSimilarityAudio; `KB audio clean_range=…` prints the gates that
//    answer every on-topic question and reject every negative (off-topic and
//    chat), or that none does.
// 7. Spoken, transcribed by the app's Whisper (SttService as setup loads it),
//    corrected as the voice chat corrects a transcript
//    (correctVoiceTranscript), then searched as text at kKbMinSimilarityText:
//    every on-topic question answered, every off-topic one and every
//    everyday chat line rejected (as knowledge_config.dart states).
// 8. The self-test's clip (assets/selftest/what_time_is_it.wav) against its
//    words: the cosine kSelfTestAudioTextMinCosine is calibrated from
//    (`KB_SELFTEST_COSINE`), above it, and every unrelated pair (the clip
//    against other questions' words, other clips against its words;
//    `KB_SELFTEST_NEGATIVE`) below it.
//
// Steps 4, 6 and 7 print one JSON line per question (`KB_GOLDEN {...}`:
// path, kind (on, off, chat), the best score, the expected document's score
// and rank, answered or rejected at the gate, latency, and the transcript on
// the Whisper path).
//
// Prints `KB_EMBEDDER load=<ms>ms rss=<before>-><after> media=<m>`,
// `KB_PREBUILT install=<ms>ms device_index=<ms>ms same_top=<n>/<all>
// same_doc=<n>/<on> same_gate=<n>/<all> max_delta=<x>`, per path the gate sweep
// 0.50–0.90 (step 0.025), `KB <path> scores on=<min>..<max> off=<min>..<max>
// gap=<x>` (the expected document's score on topic, the best hit's off
// topic) and `KB <path> on=<a>/<on> off=<b>/<off> gate=<g>
// query_p50=<ms>ms`, and
// `KB index=<ms>ms chunks=<n> reuse=<ms>ms prebuilt=<ms>ms`.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart'
    show kWhisperSttConfig;
import 'package:litert_edge_demos/data/repositories/knowledge_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_digests.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/data/services/knowledge/vector_store_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/domain/audio/transcript_corrections.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart'
    show
        kSelfTestAudioQuestionAsset,
        kSelfTestAudioQuestionText,
        kSelfTestAudioTextMinCosine;
import 'package:litert_edge_demos/utils/cosine.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';
import 'package:path_provider/path_provider.dart';

import 'support/kb_golden_clips.dart';
import 'support/question_audio.dart';

typedef _Answer = ({List<Passage> top, Duration latency});

// The Whisper path's on-topic result depends on the voice chat's transcript
// corrections (kVoiceTranscriptCorrections): Whisper base writes "LiteRT" as
// "littered", which keeps on-topic questions from their documents. The
// corrections are phrase-level ("what is littered", "runs on littered",
// "littered and file", "tents or flow light"; never "littered" alone).

/// The golden set's answers from one index.
typedef _On = ({
  String q,
  String doc,
  List<Passage> top,
  Duration latency,
  String? said,
});
typedef _Off = ({String q, List<Passage> top, Duration latency, String? said});
typedef _Golden = ({List<_On> on, List<_Off> off});

Future<_Answer> _ask(KnowledgeRepository knowledge, String question) =>
    _search(knowledge, KnowledgeQuery.text(question), question);

Future<_Answer> _search(
  KnowledgeRepository knowledge,
  KnowledgeQuery query,
  String label,
) async {
  switch (await knowledge.retrieve(query)) {
    case Ok(:final value):
      expect(
        value.outcome,
        isNot(RetrievalOutcome.unavailable),
        reason: 'the knowledge base must be ready: ${value.detail}',
      );
      return (top: value.candidates, latency: value.latency!);
    case Error(:final error):
      fail('retrieval failed for "$label": $error');
  }
}

Future<Uint8List> _asset(String key) async {
  final data = await rootBundle.load(key);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

/// An on-topic question is answered when its expected document is among the
/// hits at or above [gate] (the app gates each hit).
bool _answered(List<Passage> top, String doc, double gate) =>
    top.any((p) => p.doc == doc && p.similarity >= gate);

/// An off-topic question is rejected when no hit reaches [gate].
bool _rejected(List<Passage> top, double gate) =>
    top.isEmpty || top.first.similarity < gate;

double _best(List<Passage> top) => top.isEmpty ? 0 : top.first.similarity;

/// The expected document's best score among the hits (0 when absent).
double _expected(List<Passage> top, String doc) => top
    .where((p) => p.doc == doc)
    .fold(0.0, (m, p) => math.max(m, p.similarity));

double _r(double x) => double.parse(x.toStringAsFixed(4));

/// One path's answers: printed per question as JSON, swept over gates,
/// summarized; returns the counts at [gate]. [offKind]: what the negatives
/// are in the JSON lines (`off`, or `chat` for the everyday chat); with
/// [printOn] false the on-topic lines are left out (printed by an earlier
/// report of the same answers).
({int on, int off, double onMin, double offMax}) _report(
  String path,
  List<
    ({String q, String doc, List<Passage> top, Duration latency, String? said})
  >
  on,
  List<({String q, List<Passage> top, Duration latency, String? said})> off,
  double gate, {
  String offKind = 'off',
  bool printOn = true,
}) {
  for (final (i, a) in on.indexed) {
    if (!printOn) break;
    final rank = a.top.indexWhere((p) => p.doc == a.doc);
    debugPrint(
      'KB_GOLDEN ${jsonEncode({'path': path, 'kind': 'on', 'i': i + 1, 'q': a.q, 'said': ?a.said, 'expect': a.doc, 'best': _r(_best(a.top)), 'bestDoc': a.top.isEmpty ? null : a.top.first.doc, 'expectScore': _r(_expected(a.top, a.doc)), 'rank': rank < 0 ? null : rank + 1, 'gate': gate, 'answered': _answered(a.top, a.doc, gate), 'ms': a.latency.inMilliseconds})}',
    );
  }
  for (final (i, a) in off.indexed) {
    debugPrint(
      'KB_GOLDEN ${jsonEncode({'path': path, 'kind': offKind, 'i': i + 1, 'q': a.q, 'said': ?a.said, 'best': _r(_best(a.top)), 'bestDoc': a.top.isEmpty ? null : a.top.first.doc, 'gate': gate, 'rejected': _rejected(a.top, gate), 'ms': a.latency.inMilliseconds})}',
    );
  }
  debugPrint('KB $path sweep gate  on/${on.length}  off/${off.length}');
  for (var i = 0; i <= 16; i++) {
    final g = 0.50 + i * 0.025;
    final a = on.where((x) => _answered(x.top, x.doc, g)).length;
    final b = off.where((x) => _rejected(x.top, g)).length;
    debugPrint(
      'KB $path sweep ${g.toStringAsFixed(3)}  $a  $b'
      '${(g - gate).abs() < 1e-9 ? '  <- gate' : ''}',
    );
  }
  final onScores = [for (final a in on) _expected(a.top, a.doc)]..sort();
  final offScores = [for (final a in off) _best(a.top)]..sort();
  final latencies = [
    for (final a in on) a.latency,
    for (final a in off) a.latency,
  ]..sort();
  final answered = on.where((a) => _answered(a.top, a.doc, gate)).length;
  final rejected = off.where((a) => _rejected(a.top, gate)).length;
  debugPrint(
    'KB $path scores on=${onScores.first.toStringAsFixed(3)}..'
    '${onScores.last.toStringAsFixed(3)} '
    'off=${offScores.first.toStringAsFixed(3)}..'
    '${offScores.last.toStringAsFixed(3)} '
    'gap=${(onScores.first - offScores.last).toStringAsFixed(3)}',
  );
  debugPrint(
    'KB $path on=$answered/${on.length} off=$rejected/${off.length} '
    'gate=$gate query_p50=${latencies[latencies.length ~/ 2].inMilliseconds}ms '
    'first=${(on.isEmpty ? off.first.latency : on.first.latency).inMilliseconds}ms',
  );
  return (
    on: answered,
    off: rejected,
    onMin: onScores.first,
    offMax: offScores.last,
  );
}

String _fmt(List<Passage> top) =>
    top.map((p) => '${p.doc}(${p.similarity.toStringAsFixed(3)})').join(', ');

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('KB: index, reuse, and the golden set clears the gate', (
    tester,
  ) async {
    // The file built into the app, as setup finds it (in place on desktop
    // and iOS, extracted and verified on Android).
    final EmbedderSource source;
    switch (await BundledModelFiles().pathOf(kBundledEmbedderModel)) {
      case Ok(value: final path):
        source = EmbedderFromFiles(modelPath: path);
        debugPrint('KB_SOURCE bundled $path');
      case Error(:final error):
        fail('The built-in EmbeddingGemma: $error');
    }
    await initEdgeAi();

    // The embedder, exactly as setup loads it: its audio encoder is warmed
    // up with the self-test's clip (so the first spoken golden question
    // costs what the others do).
    final embedder = EmbedderService(
      warmUpAudio: () => _asset(kKbAudioWarmUpAsset),
    );
    final installed = await embedder.install(source, onProgress: (_) {});
    expect(installed, isA<Ok<String>>(), reason: '$installed');
    final EmbedderInfo info;
    switch (await embedder.load()) {
      case Ok(:final value):
        info = value;
      case Error(:final error):
        fail('embedder load failed: $error');
    }
    final warmUp = await embedder.warmUp();
    expect(warmUp, isA<Ok<Duration>>());
    debugPrint(
      'KB_EMBEDDER load=${info.loadTime.inMilliseconds}ms '
      'warm_up=${(warmUp as Ok<Duration>).value.inMilliseconds}ms '
      'rss=${((info.rssBeforeBytes ?? 0) / 1048576).round()}MB->'
      '${((info.rssAfterBytes ?? 0) / 1048576).round()}MB '
      'media=${info.modalities.map((m) => m.name).join('+')} '
      'maxInputTokens=${info.maxInputTokens}',
    );
    final models = ValueNotifier<Map<ModelId, ModelState>>({
      ModelId.embeddingGemma: ModelReady(
        LoadedModelInfo(
          modelId: info.modelId,
          backend: info.backend.name,
          loadTime: info.loadTime,
          warmUpTime: Duration.zero,
        ),
      ),
    });
    // One store per launch, as in the app: a repository's close closes its
    // store for good (a closed store refuses to open again).
    VectorStoreService newStore() =>
        VectorStoreService(embedQuery: embedder.embedQuery);
    final dir = Directory(
      '${(await getApplicationSupportDirectory()).path}/kb_test',
    );
    if (dir.existsSync()) dir.deleteSync(recursive: true);

    // 1. A fresh index embedded on this device (the prebuilt one is kept out
    //    of it: this is the reference the prebuilt index is compared to).
    final first = KnowledgeRepository(
      embedder: embedder,
      store: newStore(),
      models: models,
      indexDir: () async => dir,
      prebuilt: const NoPrebuiltKbIndex('kb_retrieval_test: the reference'),
      digests: FileEmbedderDigests(cacheDir: () async => dir),
      documents: AssetKbDocumentSource(),
    );
    final KnowledgeReady indexed;
    switch (await first.ensureIndexed()) {
      case Ok(:final value):
        indexed = value;
      case Error(:final error):
        fail('indexing failed: $error');
    }
    expect(indexed.reused, isFalse);
    expect(indexed.origin, KnowledgeOrigin.device);
    expect(File('${dir.path}/index.json').existsSync(), isTrue);
    await first.close();

    // 2. The next launch: same files, same model.
    final device = KnowledgeRepository(
      embedder: embedder,
      store: newStore(),
      models: models,
      indexDir: () async => dir,
      digests: FileEmbedderDigests(cacheDir: () async => dir),
      documents: AssetKbDocumentSource(),
      prebuilt: AssetPrebuiltKbIndex(),
    );
    final KnowledgeReady reused;
    switch (await device.ensureIndexed()) {
      case Ok(:final value):
        reused = value;
      case Error(:final error):
        fail('reopening the index failed: $error');
    }
    expect(reused.reused, isTrue);
    expect(reused.chunks, indexed.chunks);

    // The golden set on the on-device index: the reference.
    final golden = parseKbGolden(
      jsonDecode(await rootBundle.loadString('test_assets/kb_golden.json')),
    );
    final onTopic = golden.onTopic;
    final offTopic = golden.offTopic;
    final everydayChat = golden.everydayChat;

    // The golden questions spoken by the app's own TTS, into memory (the
    // clips are not shipped): 16 kHz mono PCM16 WAVs, as a capture is handed
    // to the knowledge base, each within Whisper's window (EmbeddingGemma 2's
    // audio cap too).
    final speakWatch = Stopwatch()..start();
    final tts = await loadAppTts();
    final spokenOnClips = <Uint8List>[];
    final spokenOffClips = <Uint8List>[];
    final spokenChatClips = <Uint8List>[];
    for (final (list, texts) in [
      (spokenOnClips, [for (final (:q, doc: _) in onTopic) q]),
      (spokenOffClips, offTopic),
      (spokenChatClips, everydayChat),
    ]) {
      for (final text in texts) {
        final clip = await speakQuestion(
          tts,
          text,
          window: kWhisperSttConfig.window,
        );
        list.add(wavFromPcm16(clip.pcm, sampleRate: kClipSampleRate));
      }
    }
    await tts.close();
    debugPrint(
      'KB_SPOKEN ${spokenOnClips.length + spokenOffClips.length} golden '
      'questions and ${spokenChatClips.length} lines of everyday chat spoken '
      'by Inflect in ${speakWatch.elapsedMilliseconds}ms',
    );
    Future<_Golden> answer(KnowledgeRepository knowledge) async {
      final on = <_On>[];
      for (final (:q, :doc) in onTopic) {
        final (:top, :latency) = await _ask(knowledge, q);
        on.add((q: q, doc: doc, top: top, latency: latency, said: null));
      }
      final off = <_Off>[];
      for (final q in offTopic) {
        final (:top, :latency) = await _ask(knowledge, q);
        off.add((q: q, top: top, latency: latency, said: null));
      }
      return (on: on, off: off);
    }

    final onDevice = await answer(device);
    await device.close();

    // 3. The prebuilt index built into the app (assets/kb_index), installed
    //    into an empty directory as on a first launch: nothing embedded.
    final prebuiltDir = Directory('${dir.path}_prebuilt');
    if (prebuiltDir.existsSync()) prebuiltDir.deleteSync(recursive: true);
    final knowledge = KnowledgeRepository(
      embedder: embedder,
      store: newStore(),
      models: models,
      indexDir: () async => prebuiltDir,
      digests: FileEmbedderDigests(cacheDir: () async => prebuiltDir),
      documents: AssetKbDocumentSource(),
      prebuilt: AssetPrebuiltKbIndex(),
    );
    final KnowledgeReady fromPrebuilt;
    switch (await knowledge.ensureIndexed()) {
      case Ok(:final value):
        fromPrebuilt = value;
      case Error(:final error):
        fail('installing the prebuilt index failed: $error');
    }
    expect(
      fromPrebuilt.origin,
      KnowledgeOrigin.prebuilt,
      reason: 'prebuilt index not used: ${fromPrebuilt.prebuiltSkipped}',
    );
    expect(fromPrebuilt.reused, isFalse);
    expect(fromPrebuilt.chunks, indexed.chunks);
    final prebuilt = await answer(knowledge);

    // The same answers: the prebuilt vectors were embedded on the build
    // machine's CPU, these on this device's, so near-equal chunks may swap
    // places on another architecture. What the app does with the result must
    // not change: the same top document for on-topic questions and the same
    // gate decision for every question, with scores within 1e-2.
    bool gateOn(List<Passage> top) =>
        top.isNotEmpty && top.first.similarity >= kKbMinSimilarityText;
    final pairs = [
      for (var i = 0; i < onTopic.length; i++)
        (
          q: onTopic[i].q,
          a: onDevice.on[i].top,
          b: prebuilt.on[i].top,
          on: true,
        ),
      for (var i = 0; i < offTopic.length; i++)
        (
          q: offTopic[i],
          a: onDevice.off[i].top,
          b: prebuilt.off[i].top,
          on: false,
        ),
    ];
    var sameTop = 0;
    var sameDoc = 0;
    var sameGate = 0;
    var maxDelta = 0.0;
    final problems = <String>[];
    for (final (:q, :a, :b, :on) in pairs) {
      final identical =
          a.length == b.length &&
          [for (var i = 0; i < a.length; i++) a[i].id == b[i].id]
              .every((x) => x);
      if (identical) sameTop++;
      // Scores: the top hit's (what the gate sees), and every chunk both
      // lists hold.
      if (a.isNotEmpty && b.isNotEmpty) {
        maxDelta = math.max(
          maxDelta,
          (a.first.similarity - b.first.similarity).abs(),
        );
      }
      final scores = {for (final p in b) p.id: p.similarity};
      for (final p in a) {
        if (scores[p.id] case final other?) {
          maxDelta = math.max(maxDelta, (p.similarity - other).abs());
        }
      }
      final docAgrees =
          a.isNotEmpty && b.isNotEmpty && a.first.doc == b.first.doc;
      if (on && docAgrees) sameDoc++;
      final gateAgrees = gateOn(a) == gateOn(b);
      if (gateAgrees) sameGate++;
      if (!identical) {
        debugPrint(
          'KB_PREBUILT differs q="$q" device=[${_fmt(a)}] prebuilt=[${_fmt(b)}]',
        );
      }
      if ((on && !docAgrees) || !gateAgrees) {
        problems.add(
          '${on && !docAgrees ? 'top document' : 'gate'} differs: q="$q"',
        );
      }
    }
    debugPrint(
      'KB_PREBUILT install=${fromPrebuilt.elapsed.inMilliseconds}ms '
      'device_index=${indexed.elapsed.inMilliseconds}ms '
      'same_top=$sameTop/${pairs.length} '
      'same_doc=$sameDoc/${onTopic.length} '
      'same_gate=$sameGate/${pairs.length} '
      'max_delta=${maxDelta.toStringAsExponential(2)}',
    );
    expect(problems, isEmpty, reason: 'the prebuilt index answers differently');
    expect(maxDelta, lessThanOrEqualTo(1e-2), reason: 'scores within 1e-2');

    // 4. Typed, on the prebuilt index (what the app uses): an on-topic
    //    question counts only when its document is among the gated hits.
    for (final a in prebuilt.on) {
      debugPrint(
        '${_answered(a.top, a.doc, kKbMinSimilarityText) ? 'ok  ' : 'MISS'} '
        'on  expect=${a.doc} top=[${_fmt(a.top)}] q="${a.q}"',
      );
    }
    final text = _report(
      'text',
      prebuilt.on,
      prebuilt.off,
      kKbMinSimilarityText,
    );

    // 5. The skill trigger phrases' best matches, in the golden file's shape.
    final triggers = jsonDecode(
      await rootBundle.loadString('test_assets/skill_trigger_similarity.json'),
    ) as Map<String, Object?>;
    final measured = <Map<String, Object?>>[];
    for (final row in triggers['phrases']! as List<Object?>) {
      if (row case {'skill': final String skill, 'q': final String q}) {
        final (top: hits, latency: _) = await _ask(knowledge, q);
        measured.add({
          'skill': skill,
          'q': q,
          'top': double.parse(
            (hits.isEmpty ? 0.0 : hits.first.similarity).toStringAsFixed(3),
          ),
          'doc': hits.isEmpty ? null : hits.first.id,
        });
      } else {
        throw FormatException('Unexpected trigger entry', row);
      }
    }
    debugPrint(
      'KB_SKILL_TRIGGERS ${jsonEncode({'measured': '${info.modelId} over the ${indexed.chunks} '
          'knowledge-base chunks: top cosine similarity per trigger phrase', 'gate': kKbMinSimilarityText, 'phrases': measured})}',
    );

    // 6. Spoken, embedded as audio: each golden question's WAV as it is.
    final spokenOn = <_On>[];
    for (final (i, (:q, :doc)) in onTopic.indexed) {
      final (:top, :latency) = await _search(
        knowledge,
        KnowledgeQuery.audio(spokenOnClips[i]),
        'spoken on_${i + 1}',
      );
      spokenOn.add((q: q, doc: doc, top: top, latency: latency, said: null));
    }
    final spokenOff = <_Off>[];
    for (final (i, q) in offTopic.indexed) {
      final (:top, :latency) = await _search(
        knowledge,
        KnowledgeQuery.audio(spokenOffClips[i]),
        'spoken off_${i + 1}',
      );
      spokenOff.add((q: q, top: top, latency: latency, said: null));
    }
    final audio = _report('audio', spokenOn, spokenOff, kKbMinSimilarityAudio);
    // The everyday chat, spoken: no line may clear the audio gate.
    final spokenChat = <_Off>[];
    for (final (i, q) in everydayChat.indexed) {
      final (:top, :latency) = await _search(
        knowledge,
        KnowledgeQuery.audio(spokenChatClips[i]),
        'spoken chat_${i + 1}',
      );
      spokenChat.add((q: q, top: top, latency: latency, said: null));
    }
    final audioChat = _report(
      'audio_chat',
      spokenOn,
      spokenChat,
      kKbMinSimilarityAudio,
      offKind: 'chat',
      printOn: false,
    );
    // The gates that answer every on-topic question and reject every
    // negative: (highest negative, lowest on-topic expected score].
    final audioNegMax = math.max(audio.offMax, audioChat.offMax);
    debugPrint(
      audio.onMin > audioNegMax
          ? 'KB audio clean_range=(${audioNegMax.toStringAsFixed(4)}, '
                '${audio.onMin.toStringAsFixed(4)}] '
                'middle=${((audioNegMax + audio.onMin) / 2).toStringAsFixed(4)} '
                'gate=$kKbMinSimilarityAudio'
          : 'KB audio clean_range=none: the highest negative '
                '${audioNegMax.toStringAsFixed(4)} reaches the lowest on-topic '
                'score ${audio.onMin.toStringAsFixed(4)}',
    );
    final clipSeconds = [
      for (final wav in [
        ...spokenOnClips,
        ...spokenOffClips,
        ...spokenChatClips,
      ])
        wavDuration(wav).inMilliseconds / 1000,
    ]..sort();
    debugPrint(
      'KB audio clips=${clipSeconds.length} '
      'duration=${clipSeconds.first.toStringAsFixed(2)}..'
      '${clipSeconds.last.toStringAsFixed(2)}s',
    );

    // 7. Spoken, transcribed by the app's Whisper, searched as text.
    final whisperFiles = kBundledSttFiles[ModelId.whisperBase]!;
    final bundled = BundledModelFiles();
    final whisperModel = await bundled.pathOf(whisperFiles.model);
    final whisperTokenizer = await bundled.pathOf(whisperFiles.tokenizer);
    final stt = SttService();
    expect(
      await stt.install(
        ModelId.whisperBase,
        source: SttFromFiles(
          modelPath: (whisperModel as Ok<String>).value,
          tokenizerPath: (whisperTokenizer as Ok<String>).value,
        ),
        onProgress: (_) {},
      ),
      isA<Ok<String>>(),
    );
    expect(await stt.load(ModelId.whisperBase), isA<Ok<Duration>>());
    // What the voice chat searches with: Whisper's words, corrected.
    var corrected = 0;
    Future<String> transcribe(String label, Uint8List wav) async {
      final pcm = Uint8List.fromList(pcm16FromWav(wav));
      final watch = Stopwatch()..start();
      final heard = await stt.transcribe(pcm, expected: ModelId.whisperBase);
      final said = correctVoiceTranscript(heard);
      if (said != heard) corrected++;
      debugPrint(
        'KB_WHISPER ${watch.elapsedMilliseconds}ms $label "$heard"'
        '${said == heard ? '' : ' -> "$said"'}',
      );
      return said;
    }

    final heardOn = <_On>[];
    for (final (i, (:q, :doc)) in onTopic.indexed) {
      final said = await transcribe('on_${i + 1}', spokenOnClips[i]);
      final (:top, :latency) = await _ask(knowledge, said);
      heardOn.add((q: q, doc: doc, top: top, latency: latency, said: said));
    }
    final heardOff = <_Off>[];
    for (final (i, q) in offTopic.indexed) {
      final said = await transcribe('off_${i + 1}', spokenOffClips[i]);
      final (:top, :latency) = await _ask(knowledge, said);
      heardOff.add((q: q, top: top, latency: latency, said: said));
    }
    final heardChat = <_Off>[];
    for (final (i, q) in everydayChat.indexed) {
      final said = await transcribe('chat_${i + 1}', spokenChatClips[i]);
      final (:top, :latency) = await _ask(knowledge, said);
      heardChat.add((q: q, top: top, latency: latency, said: said));
    }
    debugPrint(
      'KB whisper corrected=$corrected/'
      '${onTopic.length + offTopic.length + everydayChat.length}',
    );
    await stt.close();
    final whisper = _report('whisper', heardOn, heardOff, kKbMinSimilarityText);
    final whisperChat = _report(
      'whisper_chat',
      heardOn,
      heardChat,
      kKbMinSimilarityText,
      offKind: 'chat',
      printOn: false,
    );

    // 8. The self-test's clip against its words, mapped as the knowledge
    //    base maps a question.
    final selfTestClip = await _asset(kSelfTestAudioQuestionAsset);
    final spoken = await embedder.embedQuery(
      KnowledgeQuery.audio(selfTestClip),
    );
    final written = await embedder.embedQuery(
      const KnowledgeQuery.text(kSelfTestAudioQuestionText),
    );
    final spokenVector = (spoken as Ok<List<double>>).value;
    final writtenVector = (written as Ok<List<double>>).value;
    final selfTestCosine = cosineSimilarity(spokenVector, writtenVector);
    debugPrint(
      'KB_SELFTEST_COSINE ${selfTestCosine.toStringAsFixed(4)} '
      'clip=${wavDuration(selfTestClip).inMilliseconds}ms '
      'gate=$kSelfTestAudioTextMinCosine',
    );
    // Negative controls: the clip against other questions' words, and other
    // questions' clips against its words. A broken audio path scores like
    // these; the gate must sit above every one of them.
    final negatives = <double>[];
    for (final i in [0, 6, 11]) {
      final other = (await embedder.embedQuery(
        KnowledgeQuery.text(offTopic[i]),
      ) as Ok<List<double>>).value;
      final otherClip = (await embedder.embedQuery(
        KnowledgeQuery.audio(spokenOffClips[i]),
      ) as Ok<List<double>>).value;
      final clipVsText = cosineSimilarity(spokenVector, other);
      final textVsClip = cosineSimilarity(otherClip, writtenVector);
      negatives.addAll([clipVsText, textVsClip]);
      debugPrint(
        'KB_SELFTEST_NEGATIVE clip_vs="${offTopic[i]}" '
        '${clipVsText.toStringAsFixed(4)} '
        '"${offTopic[i]}"_clip_vs_text ${textVsClip.toStringAsFixed(4)}',
      );
    }

    debugPrint(
      'KB index=${indexed.elapsed.inMilliseconds}ms '
      'chunks=${indexed.chunks} reuse=${reused.elapsed.inMilliseconds}ms '
      'prebuilt=${fromPrebuilt.elapsed.inMilliseconds}ms '
      'backend=${info.backend.name} dim=${info.dimension}',
    );

    expect(prebuilt.on, hasLength(19));
    expect(prebuilt.off, hasLength(12));
    // What lib/config/knowledge_config.dart states: every golden question on
    // its side of the gate.
    expect(text.on, 19, reason: 'typed on-topic');
    expect(text.off, 12, reason: 'typed off-topic');
    expect(audio.on, 19, reason: 'spoken on-topic');
    expect(audio.off, 12, reason: 'spoken off-topic');
    expect(
      audioChat.off,
      everydayChat.length,
      reason:
          'spoken everyday chat below the audio gate: '
          '${[for (final a in spokenChat) '"${a.q}" ${_best(a.top).toStringAsFixed(4)}'].join(', ')}',
    );
    expect(whisper.on, 19, reason: 'Whisper on-topic');
    expect(whisper.off, 12, reason: 'Whisper off-topic');
    expect(
      whisperChat.off,
      everydayChat.length,
      reason:
          'everyday chat through Whisper below the text gate: '
          '${[for (final a in heardChat) '"${a.said}" ${_best(a.top).toStringAsFixed(4)}'].join(', ')}',
    );
    expect(
      selfTestCosine,
      greaterThanOrEqualTo(kSelfTestAudioTextMinCosine),
      reason: 'the self-test\'s audio ↔ text gate',
    );
    expect(
      negatives.every((n) => n < kSelfTestAudioTextMinCosine),
      isTrue,
      reason: 'an unrelated pair must fail the self-test\'s gate: $negatives',
    );

    await knowledge.close();
    models.dispose();
    await embedder.close();
  }, timeout: const Timeout(Duration(minutes: 15)));
}
