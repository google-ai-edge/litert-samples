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

// Demo 3's visual memory on a real target (macOS first), without a window:
// the built-in YOLO26n in its worker isolate on the GPU, the built-in
// EmbeddingGemma 2 740m on the CPU loaded as setup loads it (audio declared)
// and widened to images as the camera demo widens it (70 vision tokens),
// the real LiveDetectionRepository playing a FixtureFrameSource sequence,
// and the real VisualMemoryRepository sampling it. The questions are typed (the router's phrase, as moonshine's text would
// give it); integration_test/visual_memory_app_test.dart asks them aloud in
// the app.
//
//   flutter test integration_test/visual_memory_test.dart -d macos \
//     --dart-define=MEMORY_FIXTURE_DIR=<a copy of test_assets/showcase>
//
// The sandboxed macOS debug build reads files only from its container and
// ~/Downloads: pass a copy of test_assets/showcase under ~/Downloads. The
// cats photo and the GATE 42 sign are app assets. No window is needed (no
// frame is pumped), so it also runs with the screen locked. For honest
// timings keep the test app out of App Nap:
//   defaults write com.google.ai.edge.examples.litertEdgeDemos NSAppSleepDisabled -bool YES
//
// The RSS figures (`MEMORY_WIDEN`, `MEMORY_RSS`) are measurements for the
// spec's resource table, not gates: nothing is asserted on them. The one
// limit the spec sets is asserted: a frame embed within 3 s
// (docs/design/eg2-multimodal.md, "Resources: go / no-go").
//
// 0. The image encoder added to setup's embedder (`MEMORY_WIDEN`: the
//    reload's time and the process RSS before and after, measured), with a
//    knowledge-base question asked during the reload: it waits and succeeds.
// 1. Self-test step 7c's pair: the bundled cats photo, made into a memory
//    frame as the worker makes one (memoryFramePng: ≤ 320 px, filter-0
//    PNG), against "two cats", and against unrelated captions
//    (`MEMORY_SELFTEST cos=… negatives=…`; the photo as it is for
//    comparison): the pair clears kSelfTestImageTextMinCosine, every
//    negative stays below it. The process RSS before and after the first
//    image (`MEMORY_RSS`, measured) and its time (within 3 s).
// 2. One pass of five scenes (the Wi-Fi password note, the street sign, the
//    pirate mug, the cats, the GATE 42 sign), each held 6 s: the sampler
//    stores one frame per scene (`MEMORY_ENTRY …`: its PNG size, worker and
//    embed times; the last frame embed within 3 s), then sampling stops.
// 3. Memory questions through the router's phrase and
//    VisualMemoryRepository.recall: the scene questions find their frame,
//    the questions about things never shown are rejected
//    (`MEMORY_Q {...}`: every scene's cosine, the verdict). `MEMORY_GATE`
//    prints the best positive / negative cosines and the margins the
//    acceptance constants are set from.
//    Questions that fit more than one scene (`MEMORY_AMBIGUOUS`): when the
//    verdict is ambiguous, the frame answered with is the newest of the
//    competing ones.
// 4. Two frames of one scene (a brightness change and a small shift of the
//    Wi-Fi frame) against different scenes (`MEMORY_SCENE same=… cross=…`):
//    the range kMemorySameSceneCosine sits in; the Wi-Fi question's cosine
//    against those three frames (`MEMORY_TIE`): the noise kMemoryTieCosine
//    covers.
// 5. A new source session (the cats alone) clears the memory.
// 6. The cats held still for 14 s (`MEMORY_REFRESH`): one frame, its time
//    refreshed in place by the forced store, no duplicate.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/repositories/visual_memory_repository.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/detector/memory_shot.dart';
import 'package:litert_edge_demos/data/services/frames/engine_image_decoder.dart';
import 'package:litert_edge_demos/data/services/frames/fixture_frame_source.dart';
import 'package:litert_edge_demos/data/services/images/memory_frame.dart';
import 'package:litert_edge_demos/data/services/images/png_encoder.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/domain/use_cases/question_router.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart'
    show kSelfTestImageText, kSelfTestImageTextMinCosine;
import 'package:litert_edge_demos/utils/cosine.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// The spec's limit on one frame embed ("Resources: go / no-go": a device
/// over it gets the visual memory off by default).
const _frameEmbedLimit = Duration(seconds: 3);

const kMemoryFixtureDir = String.fromEnvironment('MEMORY_FIXTURE_DIR');

/// Each scene's hold: two sampler attempts at least (3 s apart).
const _hold = Duration(seconds: 6);

/// The scenes, in playing order: a name and where the image comes from (a
/// file under MEMORY_FIXTURE_DIR, or an app asset).
const _scenes = [
  ('wifi', 'sign_wifi.jpg'),
  ('street_sign', 'sign_do_not_feed.jpg'),
  ('pirate_mug', 'coco_2592_pirate_mug.jpg'),
  ('cats', 'asset:test_assets/cats.jpg'),
  ('gate42', 'asset:test_assets/gate42.png'),
];

/// Questions about what was shown, and the scene each must find.
const _positives = [
  ('Where was the sign about the wifi password?', 'wifi'),
  ('Where was the yellow street sign?', 'street_sign'),
  ('Where did I leave my pirate mug?', 'pirate_mug'),
  ('Have you seen the cats?', 'cats'),
  ('When did you last see the gate number?', 'gate42'),
];

/// Questions that fit several scenes (three signs): printed with their
/// verdict; an ambiguous one must answer with the newest competing frame.
const _ambiguousProbes = [
  'Where was the sign?',
  'Have you seen a sign?',
  'Did you see some text?',
];

/// Questions about things never shown: each must be rejected.
const _negatives = [
  'Have you seen a giraffe?',
  'Did you see a bicycle?',
  'Where is my passport?',
  'Have you seen a pizza?',
  'Did you see an elephant?',
  'Where are my car keys?',
];

/// Captions the cats photo must not match as well as its own (step 7c).
const _selfTestNegatives = [
  'a red bicycle',
  'a yellow taxi',
  'a bowl of soup',
  'a laptop on a desk',
  'a street sign',
];

Future<Uint8List> _asset(String key) async {
  final data = await rootBundle.load(key);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

String _f3(double v) => v.toStringAsFixed(3);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('visual memory: real frames, the real embedder, the right '
      'frame found, an unseen object rejected, cleared on a new source', (
    tester,
  ) async {
    if (kMemoryFixtureDir.isEmpty) {
      fail(
        'Pass --dart-define=MEMORY_FIXTURE_DIR=<a copy of '
        'test_assets/showcase under ~/Downloads>',
      );
    }
    // The scenes as files the fixture source can read (the container's
    // tmp), and their bytes.
    final fixtures = Directory(
      '${Directory.systemTemp.path}/visual_memory_fixtures',
    );
    if (fixtures.existsSync()) fixtures.deleteSync(recursive: true);
    fixtures.createSync(recursive: true);
    final paths = <String>[];
    final bytesOf = <String, Uint8List>{};
    for (final (i, (name, from)) in _scenes.indexed) {
      final bytes = from.startsWith('asset:')
          ? await _asset(from.substring('asset:'.length))
          : await File('$kMemoryFixtureDir/$from').readAsBytes();
      final ext = from.endsWith('.png') ? 'png' : 'jpg';
      // Sorted names: the fixture source plays a directory in name order.
      final path = '${fixtures.path}/${i}_$name.$ext';
      await File(path).writeAsBytes(bytes);
      paths.add(path);
      bytesOf[name] = bytes;
    }

    // The embedder, exactly as setup loads it.
    final EmbedderSource source;
    switch (await BundledModelFiles().pathOf(kBundledEmbedderModel)) {
      case Ok(value: final path):
        source = EmbedderFromFiles(modelPath: path);
      case Error(:final error):
        fail('The built-in EmbeddingGemma: $error');
    }
    await initEdgeAi();
    final embedder = EmbedderService();
    expect(
      await embedder.install(source, onProgress: (_) {}),
      isA<Ok<String>>(),
    );
    final info = switch (await embedder.load()) {
      Ok(:final value) => value,
      Error(:final error) => fail('embedder load failed: $error'),
    };
    expect(await embedder.warmUp(), isA<Ok<Duration>>());
    debugPrint(
      'MEMORY_EMBEDDER load=${info.loadTime.inMilliseconds}ms '
      'media=${info.modalities.map((m) => m.name).join('+')} '
      'rss=${(info.rssAfterBytes ?? 0) >> 20}MB',
    );
    expect(
      embedder.imageUnavailableReason,
      'EmbeddingGemma was loaded without images',
      reason: 'setup declares audio only',
    );

    // 0. The camera demo's first start adds the image encoder; a
    //    knowledge-base question asked meanwhile waits and succeeds.
    final rssBeforeWiden = ProcessInfo.currentRss;
    final widening = embedder.enableImages();
    final kbWatch = Stopwatch()..start();
    final kb = await embedder.embedQuery(
      const KnowledgeQuery.text('What is LiteRT?'),
    );
    final kbWaited = kbWatch.elapsed;
    final widened = switch (await widening) {
      Ok(:final value) => value,
      Error(:final error) => fail('enableImages: $error'),
    };
    final rssAfterWiden = ProcessInfo.currentRss;
    expect(kb, isA<Ok<List<double>>>(), reason: 'the knowledge base waits');
    expect(embedder.imageUnavailableReason, isNull);
    debugPrint(
      'MEMORY_WIDEN reload=${widened.loadTime.inMilliseconds}ms '
      'rss_before=${rssBeforeWiden >> 20}MB rss_after=${rssAfterWiden >> 20}MB '
      'delta=${(rssAfterWiden - rssBeforeWiden) >> 20}MB '
      'service_rss=${(widened.rssBeforeBytes ?? 0) >> 20}->'
      '${(widened.rssAfterBytes ?? 0) >> 20}MB '
      'kb_query_during=ok waited=${kbWaited.inMilliseconds}ms '
      'media=${embedder.modalities.map((m) => m.name).join('+')}',
    );

    Future<List<double>> image(Uint8List encoded) async =>
        switch (await embedder.embedImage(encoded)) {
          Ok(:final value) => value,
          Error(:final error) => fail('embedImage: $error'),
        };
    Future<List<double>> text(String phrase) async =>
        switch (await embedder.embedText(phrase)) {
          Ok(:final value) => value,
          Error(:final error) => fail('embedText: $error'),
        };

    // 1. Self-test 7c's pair (the photo as a memory frame, as 7c sends
    //    it), and what the first image costs.
    final catsFrame = switch (await memoryFramePng(bytesOf['cats']!)) {
      Ok(:final value) => value,
      Error(:final error) => fail('memoryFramePng: $error'),
    };
    final rssBefore = ProcessInfo.currentRss;
    final firstWatch = Stopwatch()..start();
    final cats = await image(catsFrame);
    final firstImage = firstWatch.elapsed;
    final rssAfter = ProcessInfo.currentRss;
    debugPrint(
      'MEMORY_RSS before_first_image=${rssBefore >> 20}MB '
      'after=${rssAfter >> 20}MB delta=${(rssAfter - rssBefore) >> 20}MB '
      'first_image=${firstImage.inMilliseconds}ms',
    );
    expect(
      firstImage,
      lessThanOrEqualTo(_frameEmbedLimit),
      reason: 'the first frame embed, spec limit',
    );
    final caption = await text(kSelfTestImageText);
    final pair = cosineSimilarity(cats, caption);
    final asIs = cosineSimilarity(await image(bytesOf['cats']!), caption);
    final negatives = <String, double>{
      for (final caption in _selfTestNegatives)
        caption: cosineSimilarity(cats, await text(caption)),
    };
    final worstNegative = negatives.values.reduce(math.max);
    debugPrint(
      'MEMORY_SELFTEST cos=${_f3(pair)} (memory frame ${catsFrame.length} B; '
      'the photo as it is ${_f3(asIs)}) gate='
      '${kSelfTestImageTextMinCosine.toStringAsFixed(2)} negatives='
      '${jsonEncode(negatives.map((k, v) => MapEntry(k, _f3(v))))}',
    );
    expect(pair, greaterThanOrEqualTo(kSelfTestImageTextMinCosine));
    expect(worstNegative, lessThan(kSelfTestImageTextMinCosine));

    // 2. One pass of the scenes through the live pipeline and the memory.
    final detector = DetectorService();
    final detectorInfo = await detector.load(
      source: DetectorBytes(await loadBundledDetector()),
      backend: standardDetectorBackend(),
    );
    expect(detectorInfo, isA<Ok<DetectorInfo>>(), reason: '$detectorInfo');
    final live = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) => switch (spec) {
        FixtureSourceSpec(:final paths) => Result.ok(
          FixtureFrameSource(paths: paths, hold: _hold),
        ),
        _ => const Result.error(
          FrameSourceUnavailableException('fixtures only'),
        ),
      },
    );
    var sampling = true;
    final memory = VisualMemoryRepository(
      capture: live.captureMemory,
      capturePending: () => live.capturePending,
      sourceStarts: live.sourceStarts,
      blackFrames: live.blackFrames,
      liveState: live.state,
      embedder: embedder,
      enableImages: embedder.enableImages,
    );
    final owner = Object();
    memory.start(owner: owner, idle: () => sampling);
    final started = await live.start(
      FixtureSourceSpec([fixtures.path]),
      owner: owner,
    );
    expect(started, isA<Ok<FrameSourceInfo>>(), reason: '$started');
    // One pass, then the sampler stops (the memory is kept).
    await Future<void>.delayed(_hold * _scenes.length);
    sampling = false;
    await Future<void>.delayed(const Duration(seconds: 2)); // the last embed
    expect(live.state.value, isA<LiveRunning>());

    // Which scene each frame shows: the nearest scene thumbnail, made from
    // the image the way the worker makes it.
    final sceneGrey = <String, Uint8List>{};
    for (final (name, _) in _scenes) {
      final decoded = await decodeEncodedImage(
        bytesOf[name]!,
        maxSide: kFixtureMaxSide,
      );
      final w = decoded.image.width;
      final h = decoded.image.height;
      decoded.image.dispose();
      final (tw, th) = fitLongSide(w, h, kMemoryThumbMaxSide);
      final rgb = downscaleRgb(
        RgbaPixels(width: w, height: h, bytes: decoded.rgba),
        tw,
        th,
      );
      sceneGrey[name] = greyThumbnail(
        rgb,
        tw,
        th,
        kMemoryGreySide * kMemoryCellPixels,
      );
    }
    String sceneOf(MemoryEntry e) => sceneGrey.entries
        .reduce(
          (a, b) =>
              greyDifference(a.value, e.grey) <= greyDifference(b.value, e.grey)
              ? a
              : b,
        )
        .key;

    final entries = memory.entries;
    final status = memory.status.value;
    for (final e in entries) {
      debugPrint(
        'MEMORY_ENTRY #${e.frameId} ${sceneOf(e)} ${e.width}x${e.height} '
        'png=${e.png.length}B counts=${e.counts}',
      );
    }
    debugPrint(
      'MEMORY_STORE frames=${status.frames} bytes=${status.bytes} '
      'last_embed=${status.lastEmbed?.inMilliseconds}ms '
      'failures=${status.failures} span=${status.span.inSeconds}s',
    );
    final seen = {for (final e in entries) sceneOf(e)};
    expect(seen, {
      for (final (name, _) in _scenes) name,
    }, reason: 'one frame per scene at least');
    expect(status.failures, 0);
    expect(status.unavailable, isNull);
    expect(status.lastEmbed, isNotNull);
    expect(
      status.lastEmbed!,
      lessThanOrEqualTo(_frameEmbedLimit),
      reason: 'a frame embed, spec limit',
    );

    // 3. Questions through the router's phrase.
    const router = QuestionRouter();
    final positiveBest = <double>[];
    final positiveMargins = <double>[];
    final negativeBest = <double>[];
    Future<MemoryRecall> ask(String question) async {
      final route = router.classify(question);
      expect(route, isA<MemoryRoute>(), reason: question);
      final phrase = (route as MemoryRoute).query;
      final query = await text(phrase);
      final perScene = <String, double>{};
      for (final e in entries) {
        final c = cosineSimilarity(query, e.vector);
        final s = sceneOf(e);
        perScene[s] = math.max(perScene[s] ?? -1, c);
      }
      final recall = switch (await memory.recall(phrase)) {
        Ok(:final value) => value,
        Error(:final error) => fail('recall "$phrase": $error'),
      };
      debugPrint(
        'MEMORY_Q ${jsonEncode({'q': question, 'phrase': phrase, 'scenes': perScene.map((k, v) => MapEntry(k, _f3(v))), 'verdict': switch (recall) {
          MemoryHit(:final entry, :final cosine, :final rival, :final ambiguous) => '${ambiguous ? 'ambiguous ' : ''}hit ${sceneOf(entry)} ${cosine == null ? 'latest' : _f3(cosine)} rival ${rival == null ? '-' : _f3(rival)}',
          MemoryMiss(:final cosine, :final reason) => 'miss ${_f3(cosine)} ($reason)',
          MemoryEmpty() => 'empty',
          MemoryDisabled() => 'disabled',
          MemoryOff(:final reason) => 'off $reason',
        }})}',
      );
      return recall;
    }

    for (final (question, scene) in _positives) {
      final recall = await ask(question);
      expect(recall, isA<MemoryHit>(), reason: question);
      final hit = recall as MemoryHit;
      expect(hit.ambiguous, isFalse, reason: question);
      expect(sceneOf(hit.entry), scene, reason: question);
      final cosine = hit.cosine!; // a search result, never the latest frame
      positiveBest.add(cosine);
      if (hit.rival case final rival?) positiveMargins.add(cosine - rival);
    }
    for (final question in _negatives) {
      final recall = await ask(question);
      expect(recall, isA<MemoryMiss>(), reason: question);
      negativeBest.add((recall as MemoryMiss).cosine);
    }
    debugPrint(
      'MEMORY_GATE positives=${_f3(positiveBest.reduce(math.min))}..'
      '${_f3(positiveBest.reduce(math.max))} negatives='
      '${_f3(negativeBest.reduce(math.min))}..'
      '${_f3(negativeBest.reduce(math.max))} min_margin='
      '${positiveMargins.isEmpty ? '-' : _f3(positiveMargins.reduce(math.min))} '
      'floor=$kMemoryMinCosine margin=$kMemoryMinMargin',
    );

    // Questions that fit several scenes: an ambiguous verdict answers with
    // the newest of the competing frames.
    for (final question in _ambiguousProbes) {
      final recall = await ask(question);
      final phrase = (router.classify(question) as MemoryRoute).query;
      final query = await text(phrase);
      final cosines = [
        for (final e in entries) cosineSimilarity(query, e.vector),
      ];
      final top = cosines.reduce(math.max);
      final competing = [
        for (final (i, e) in entries.indexed)
          if (cosines[i] >= kMemoryMinCosine &&
              top - cosines[i] < kMemoryMinMargin)
            e,
      ];
      debugPrint(
        'MEMORY_AMBIGUOUS "$question" → ${switch (recall) {
          MemoryHit(:final entry, :final ambiguous) => '${ambiguous ? 'ambiguous' : 'clear'} hit ${sceneOf(entry)} #${entry.frameId}',
          MemoryMiss(:final reason) => 'miss ($reason)',
          _ => '$recall',
        }} competing=${[for (final e in competing) '${sceneOf(e)}#${e.frameId}']}',
      );
      if (recall case MemoryHit(ambiguous: true, :final entry)) {
        final newest = competing.reduce(
          (a, b) => a.atMicros >= b.atMicros ? a : b,
        );
        expect(entry.id, newest.id, reason: question);
      }
    }

    // 4. Two frames of one scene against different scenes.
    final wifi = entries.firstWhere((e) => sceneOf(e) == 'wifi');
    final decoded = await decodeEncodedImage(
      bytesOf['wifi']!,
      maxSide: kFixtureMaxSide,
    );
    final w = decoded.image.width;
    final h = decoded.image.height;
    decoded.image.dispose();
    final variants = <List<double>>[];
    Future<double> variant(Uint8List Function(Uint8List rgba) change) async {
      final rgba = change(Uint8List.fromList(decoded.rgba));
      final (tw, th) = fitLongSide(w, h, kMemoryThumbMaxSide);
      final rgb = downscaleRgb(
        RgbaPixels(width: w, height: h, bytes: rgba),
        tw,
        th,
      );
      final vector = await image(encodePngRgb(tw, th, rgb));
      variants.add(vector);
      return cosineSimilarity(wifi.vector, vector);
    }

    final darker = await variant((rgba) {
      for (var i = 0; i < rgba.length; i += 4) {
        rgba[i] = rgba[i] * 9 ~/ 10;
        rgba[i + 1] = rgba[i + 1] * 9 ~/ 10;
        rgba[i + 2] = rgba[i + 2] * 9 ~/ 10;
      }
      return rgba;
    });
    final shifted = await variant((rgba) {
      const dx = 24; // ~2 % of the width: a hand-held camera's drift
      final out = Uint8List(rgba.length);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final sx = math.min(w - 1, x + dx);
          final o = (y * w + x) * 4;
          final i = (y * w + sx) * 4;
          out[o] = rgba[i];
          out[o + 1] = rgba[i + 1];
          out[o + 2] = rgba[i + 2];
          out[o + 3] = 255;
        }
      }
      return out;
    });
    final cross = <double>[];
    final byScene = {for (final e in entries) sceneOf(e): e};
    final names = byScene.keys.toList();
    for (var i = 0; i < names.length; i++) {
      for (var j = i + 1; j < names.length; j++) {
        cross.add(
          cosineSimilarity(
            byScene[names[i]]!.vector,
            byScene[names[j]]!.vector,
          ),
        );
      }
    }
    debugPrint(
      'MEMORY_SCENE same(darker)=${_f3(darker)} same(shift 24px)='
      '${_f3(shifted)} cross=${_f3(cross.reduce(math.min))}..'
      '${_f3(cross.reduce(math.max))} threshold=$kMemorySameSceneCosine',
    );
    expect(math.min(darker, shifted), greaterThan(kMemorySameSceneCosine));
    expect(cross.reduce(math.max), lessThan(kMemorySameSceneCosine));
    // The Wi-Fi question against one scene's three frames: how far apart
    // near-identical frames score (what kMemoryTieCosine has to cover).
    final wifiQuery = await text(
      (router.classify(_positives.first.$1) as MemoryRoute).query,
    );
    final tie = [
      for (final v in [wifi.vector, ...variants])
        cosineSimilarity(wifiQuery, v),
    ];
    final spread = tie.reduce(math.max) - tie.reduce(math.min);
    debugPrint(
      'MEMORY_TIE wifi question vs its frame / darker / shifted '
      '${tie.map(_f3).join(' / ')} spread=${_f3(spread)} '
      'tie=$kMemoryTieCosine',
    );
    expect(spread, lessThanOrEqualTo(kMemoryTieCosine));

    // 5. A new source session forgets everything.
    expect(memory.status.value.frames, greaterThan(0));
    final again = await live.start(
      FixtureSourceSpec([paths[3]]), // the cats alone
      owner: owner,
    );
    expect(again, isA<Ok<FrameSourceInfo>>());
    expect(memory.entries, isEmpty);
    expect(memory.status.value.frames, 0);
    final empty = await memory.recall('cats');
    expect((empty as Ok<MemoryRecall>).value, isA<MemoryEmpty>());

    // 6. The cats held still: one frame, its time refreshed in place.
    sampling = true;
    for (var i = 0; i < 100 && memory.entries.isEmpty; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 100));
    }
    expect(memory.entries, hasLength(1));
    final firstAt = memory.entries.single.atMicros;
    final firstBytes = memory.status.value.bytes;
    // The forced store comes at the first attempt ≥ 10 s after the frame
    // (attempts about every 3.5 s): about 10.5 s.
    await Future<void>.delayed(const Duration(seconds: 14));
    sampling = false;
    await Future<void>.delayed(const Duration(seconds: 1));
    final still = memory.entries;
    debugPrint(
      'MEMORY_REFRESH frames=${still.length} '
      'refreshed_after=${((still.last.atMicros - firstAt) / 1e6).toStringAsFixed(1)}s '
      'bytes=${memory.status.value.bytes} (first $firstBytes)',
    );
    expect(still, hasLength(1), reason: 'no duplicate of a still scene');
    expect(
      still.single.atMicros - firstAt,
      greaterThanOrEqualTo(kMemoryMaxInterval.inMicroseconds),
    );
    expect(memory.status.value.bytes, firstBytes);

    memory.stop(owner: owner);
    memory.close();
    await live.stop(owner: owner);
    await live.close();
    await detector.close();
    await embedder.close();
  });
}
