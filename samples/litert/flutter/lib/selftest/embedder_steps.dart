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

import 'dart:typed_data';

import '../config/knowledge_config.dart' show kKbAudioWarmUpAsset;
import '../data/services/images/memory_frame.dart';
import '../domain/hardware/diagnostics_report.dart' show formatBytes, seconds;
import '../domain/models/knowledge_query.dart';
import '../utils/cosine.dart';
import '../utils/result.dart';
import '../utils/wav.dart';
import 'self_test_ports.dart';
import 'step_recorder.dart';

/// The spoken question step 7b embeds: the app's own TTS (Inflect-nano-v2)
/// saying [kSelfTestAudioQuestionText], as a 16 kHz mono PCM16 WAV made by
/// tool/make_question_audio.sh.
/// The same clip warms the embedder's audio encoder up at load.
const kSelfTestAudioQuestionAsset = kKbAudioWarmUpAsset;

/// The words in [kSelfTestAudioQuestionAsset].
const kSelfTestAudioQuestionText = 'What time is it?';

/// Step 7b passes when the clip and its words embed at least this close
/// (cosine), each mapped as the knowledge base maps a question (the clip as
/// audio, the words with the `retrievalQuery` prefix). Measured by
/// integration_test/kb_retrieval_test.dart on an M4 Pro Mac's CPU
/// (2026-10-13, EmbeddingGemma 2 740m): 0.8525 for this clip and its words;
/// 0.516–0.606 for unrelated pairs (the clip against other questions' words,
/// other spoken questions against its words), which is what a broken audio
/// path looks like. 0.73 is the middle of that gap (0.12 on either side). The
/// clip is fixed, so another device only moves the value by its numerics.
const kSelfTestAudioTextMinCosine = 0.73;

/// The words step 7c embeds against the bundled cats photo
/// (`test_assets/cats.jpg`: two cats on a pink sofa, two remotes).
const kSelfTestImageText = 'two cats';

/// Step 7c passes when the photo and [kSelfTestImageText] embed at least
/// this close (cosine), each embedded as the visual memory embeds a frame
/// (the photo made into a memory frame, [memoryFramePng]: ≤ 320 px, filter-0
/// PNG, 70 vision tokens) and a question (raw text). Measured by
/// integration_test/visual_memory_test.dart on an M4 Pro Mac's CPU
/// (2026-10-14, EmbeddingGemma 2 740m widened to images as the camera demo
/// widens it, `MEMORY_SELFTEST`): 0.733 for the memory frame and "two cats"
/// (the photo as it is: 0.737); 0.474–0.533 for the memory frame and
/// unrelated captions ("a red bicycle", "a yellow taxi", "a bowl of soup", "a
/// laptop on a desk", "a street sign"), which is what a broken image path
/// looks like. 0.63 is the middle of that gap (0.10 on either side). The
/// photo is fixed, so another device only moves the value by its numerics.
const kSelfTestImageTextMinCosine = 0.63;

/// Steps 7a–7c through [SelfTestEmbedder]: EmbeddingGemma 2 ready as setup
/// loads it (audio declared), then the bundled spoken question against its
/// own words, both embedded as the knowledge base embeds a question, then
/// the image encoder loaded as the camera demo loads it and the bundled
/// photo, made into a memory frame the way the detector worker makes one
/// ([memoryFramePng]: ≤ 320 px, filter-0 PNG), against its caption, as the
/// visual memory embeds a frame and a question.
final class EmbedderSteps {
  EmbedderSteps({
    required this._recorder,
    required this._embedder,
    required this._loadClip,
    required this._loadPhoto,
    this._toMemoryFrame = memoryFramePng,
    this._minCosine = kSelfTestAudioTextMinCosine,
    this._minImageCosine = kSelfTestImageTextMinCosine,
  });

  final StepRecorder _recorder;
  final SelfTestEmbedder _embedder;

  /// The bundled WAV ([kSelfTestAudioQuestionAsset] in the app).
  final Future<Result<Uint8List>> Function() _loadClip;

  /// The bundled photo (the cats in the app).
  final Future<Result<Uint8List>> Function() _loadPhoto;

  /// The photo as the visual memory stores a frame ([memoryFramePng]).
  final Future<Result<Uint8List>> Function(Uint8List encoded) _toMemoryFrame;
  final double _minCosine;
  final double _minImageCosine;

  /// Steps 7a–7c; 7b and 7c are skipped unless 7a passed.
  Future<void> run() async {
    var ready = false;
    await _recorder.run('7a', 'embedder load (audio + text)', () async {
      final (status, details) = await _prepare();
      ready = status == StepStatus.pass;
      return (status, details);
    });
    if (!ready) {
      for (final (id, name) in [
        ('7b', 'embedder audio ↔ text'),
        ('7c', 'embedder image ↔ text'),
      ]) {
        _recorder.skip(id, name, 'the embedder is not ready');
      }
      return;
    }
    await _recorder.run('7b', 'embedder audio ↔ text', _audioVsText);
    await _recorder.run('7c', 'embedder image ↔ text', _imageVsText);
  }

  Future<StepOutcome> _imageVsText() async {
    final String encoder;
    switch (await _embedder.enableImages()) {
      case Error(:final error):
        return (StepStatus.fail, ['loading the image encoder failed: $error']);
      case Ok(value: SelfTestImagesLoad(alreadyLoaded: true)):
        encoder = 'image encoder loaded before the run (the camera demo)';
      case Ok(:final value):
        encoder =
            'image encoder loaded in ${seconds(value.loadTime)} (a reload, '
            'as on the first camera visit) · rss '
            '${_mb(value.rssBeforeBytes)} → ${_mb(value.rssAfterBytes)}';
    }
    final Uint8List photo;
    switch (await _loadPhoto()) {
      case Error(:final error):
        return (StepStatus.fail, ['the bundled photo: $error']);
      case Ok(:final value):
        photo = value;
    }
    final Uint8List frame;
    switch (await _toMemoryFrame(photo)) {
      case Error(:final error):
        return (
          StepStatus.fail,
          ['the bundled photo as a memory frame: $error'],
        );
      case Ok(:final value):
        frame = value;
    }
    final imageWatch = Stopwatch()..start();
    final image = await _embedder.embedImage(frame);
    imageWatch.stop();
    final textWatch = Stopwatch()..start();
    final text = await _embedder.embedMemoryText(kSelfTestImageText);
    textWatch.stop();
    final List<double> i;
    final List<double> t;
    switch ((image, text)) {
      case (Error(:final error), _):
        return (StepStatus.fail, ['embedding the photo failed: $error']);
      case (_, Error(:final error)):
        return (StepStatus.fail, ['embedding the text failed: $error']);
      case (Ok(value: final iv), Ok(value: final tv)):
        i = iv;
        t = tv;
    }
    if (i.length != t.length ||
        !i.any((v) => v != 0) ||
        !t.any((v) => v != 0)) {
      return (
        StepStatus.fail,
        [
          'the vectors cannot be compared: ${i.length} and ${t.length} '
              'values, or one is all zeros',
        ],
      );
    }
    final cosine = cosineSimilarity(i, t);
    final pass = cosine >= _minImageCosine;
    return (
      pass ? StepStatus.pass : StepStatus.fail,
      [
        'cosine ${cosine.toStringAsFixed(3)} '
            '(${pass ? '≥' : '<'} ${_minImageCosine.toStringAsFixed(2)}) · '
            'the bundled cats photo as a memory frame '
            '(${formatBytes(frame.length)} PNG) vs "$kSelfTestImageText"',
        encoder,
        'image embed ${seconds(imageWatch.elapsed)} (the first pays for the '
            'vision tower) · text embed ${seconds(textWatch.elapsed)}',
        if (!pass)
          'the photo and its caption are not close: the visual memory would '
              'not find what it saw',
      ],
    );
  }

  Future<StepOutcome> _prepare() async {
    switch (await _embedder.prepare()) {
      case Error(:final error):
        return (StepStatus.fail, ['$error']);
      case Ok(value: final e):
        return (
          StepStatus.pass,
          [
            '${e.modelId} · ${e.backend} · ${e.media} · '
                '${e.borrowed ? 'the app\'s, loaded before the run' : 'load ${seconds(e.loadTime ?? Duration.zero)} · warm-up ${seconds(e.warmUpTime ?? Duration.zero)}'}',
          ],
        );
    }
  }

  Future<StepOutcome> _audioVsText() async {
    final Uint8List wav;
    switch (await _loadClip()) {
      case Error(:final error):
        return (StepStatus.fail, ['the bundled clip: $error']);
      case Ok(:final value):
        wav = value;
    }
    final Duration clip;
    try {
      clip = wavDuration(wav);
    } on FormatException catch (e) {
      return (StepStatus.fail, ['the bundled clip: $e']);
    }
    final audioWatch = Stopwatch()..start();
    final audio = await _embedder.embed(KnowledgeQuery.audio(wav));
    audioWatch.stop();
    final textWatch = Stopwatch()..start();
    final text = await _embedder.embed(
      const KnowledgeQuery.text(kSelfTestAudioQuestionText),
    );
    textWatch.stop();
    final List<double> a;
    final List<double> t;
    switch ((audio, text)) {
      case (Error(:final error), _):
        return (StepStatus.fail, ['embedding the clip failed: $error']);
      case (_, Error(:final error)):
        return (StepStatus.fail, ['embedding the text failed: $error']);
      case (Ok(value: final av), Ok(value: final tv)):
        a = av;
        t = tv;
    }
    if (a.length != t.length ||
        !a.any((v) => v != 0) ||
        !t.any((v) => v != 0)) {
      return (
        StepStatus.fail,
        [
          'the vectors cannot be compared: ${a.length} and ${t.length} '
              'values, or one is all zeros',
        ],
      );
    }
    final cosine = cosineSimilarity(a, t);
    final pass = cosine >= _minCosine;
    return (
      pass ? StepStatus.pass : StepStatus.fail,
      [
        'cosine ${cosine.toStringAsFixed(3)} '
            '(${pass ? '≥' : '<'} ${_minCosine.toStringAsFixed(2)}) · '
            '"$kSelfTestAudioQuestionText" spoken (${seconds(clip)}) vs '
            'written',
        'audio embed ${seconds(audioWatch.elapsed)} · text embed '
            '${seconds(textWatch.elapsed)}',
        if (!pass)
          'the spoken question and its words are not close: the audio path '
              'of the knowledge base would not find its documents',
      ],
    );
  }
}

String _mb(int? bytes) => bytes == null ? 'n/a' : formatBytes(bytes);
