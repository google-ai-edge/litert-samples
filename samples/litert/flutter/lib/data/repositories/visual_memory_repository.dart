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
import 'dart:math' as math;

import 'package:flutter/foundation.dart';

import '../../config/live_camera_config.dart';
import '../../domain/models/detection_summary.dart';
import '../../domain/models/live_state.dart';
import '../../domain/models/model_id.dart';
import '../../domain/models/model_state.dart';
import '../../domain/models/visual_memory.dart';
import '../../domain/ports/visual_memory_embedder.dart';
import '../../domain/use_cases/memory_acceptance.dart';
import '../../utils/cosine.dart';
import '../../utils/result.dart';
import '../services/detector/memory_shot.dart' show isSceneChange, sceneChange;
import 'visual_memory/memory_sampler.dart';
import 'visual_memory/memory_store.dart';

/// The live pipeline's memory capture; `LiveDetectionRepository.captureMemory`.
typedef CaptureMemoryShot = Future<Result<MemoryShot>> Function({
  Uint8List? reference,
});

/// Loads EmbeddingGemma's image encoder next to its text and audio ones;
/// `ModelRepository.enableEmbedderImages`. Never throws.
typedef EnableMemoryImages = Future<Result<void>> Function();

/// Demo 3's visual memory: frames EmbeddingGemma 2 embedded while the demo
/// was idle, searched by meaning with a spoken question's phrase.
///
/// - **Owner**: the camera screen [start]s it on entry and [stop]s it on
///   leaving; nothing is sampled without an owner, and leaving forgets
///   everything. Like live detection, a stop from anyone but the current
///   owner is a no-op.
/// - **Switch** ([setEnabled], the camera settings' "Visual memory"): off,
///   nothing is sampled, every frame is forgotten, the image encoder is
///   never loaded for it and a question gets [MemoryDisabled].
/// - **Image encoder**: setup loads EmbeddingGemma with audio only; the
///   first time an owner starts the memory (switch on, embedder ready) it
///   asks for the image encoder ([EnableMemoryImages]: one reload, during
///   which the knowledge base's embeddings wait rather than fail). Until it
///   is loaded the status says [VisualMemoryStatus.preparing] and nothing is
///   sampled; a failed load turns the memory off with the reason (the
///   knowledge base keeps its text and audio embedder) until [clear] or a
///   new owner. It stays loaded until the app exits (the embedder's media
///   only widen).
/// - **Sampler** (every [kMemoryTick]): a frame is asked for only when it is
///   due ([MemorySampler], on a monotonic clock; the frames' times and ages
///   are the wall clock's), live detection runs (not paused), the frames are
///   not black, no question capture is pending and the owner says the demo
///   is idle (no turn, no generation, no playback). A busy tick drops the
///   sample and restarts the gap, and so does the first idle tick after a
///   busy one: the next frame comes a full gap after the demo turned idle,
///   not the moment a reply drains (when a follow-up press is likeliest).
///   Nothing queues. The worker compares the frame's grey thumbnail with the
///   last stored one (re-aligned for a hand-held camera's shift of a few
///   pixels, brightness matched; a change is a mean difference of
///   `kMemorySceneChange` or `kMemoryChangedCells` cells changed by
///   `kMemoryCellChange`, so one small object counts) and makes a PNG only
///   for a scene change; an unchanged scene once [MemorySampler.forced]
///   refreshes the newest frame's last-seen time in place (no duplicate, no
///   embedding). The embedding is skipped when the
///   demo got busy meanwhile, so a question's embedding never waits behind
///   more than the one frame already in the embedder.
/// - **Store** ([MemoryStore]): 200 frames and 16 MB of thumbnails at most,
///   oldest out; cleared on leaving the demo, when a new detection session
///   starts ([sourceStarts]: a source change, Retry; not the restart after
///   the camera was released in the background, nor a detector switch) and
///   by [clear] ("Clear memory"); never written to disk.
/// - **Failures**: each failed frame embedding counts in [status] (so does
///   a vector no cosine can use, a NaN or infinite value or all zeros,
///   which is never stored: [MemoryVectorException]); after
///   [kMemoryFailuresUntilOff] in a row the memory is off with the last
///   failure as its reason, until it is cleared. An embedder that is not
///   ready (loading, failed, closed) also turns it off, with that reason. A
///   question then gets [MemoryOff].
/// - **Recall** ([recall], [judgeMemory]): one clear scene is a [MemoryHit]
///   with its latest sighting; two scenes within the margin, the best at
///   least the floor plus the margin, are an ambiguous [MemoryHit] with the
///   newest of the competing frames; nothing high enough or distinct enough
///   is a [MemoryMiss].
class VisualMemoryRepository {
  VisualMemoryRepository({
    required this._capture,
    required this._embedder,
    required this._enableImages,
    required this._sourceStarts,
    required this._blackFrames,
    required this._liveState,
    required this._capturePending,
    this._models,
    this._tickInterval = kMemoryTick,
    MemorySampler? sampler,
    MemoryStore? store,
    this._countScore = kCountScore,
    this._failuresUntilOff = kMemoryFailuresUntilOff,
    this._minCosine = kMemoryMinCosine,
    this._minMargin = kMemoryMinMargin,
    this._separation = kMemoryMarginSeparation,
    this._sameScene = kMemorySameSceneCosine,
    this._tie = kMemoryTieCosine,
    int Function()? wallClockMicros,
    int Function()? monotonicMicros,
  }) : _sampler = sampler ?? MemorySampler(),
       _store = store ?? MemoryStore(),
       _wallClock = wallClockMicros ?? _wallMicros,
       _monotonic = monotonicMicros ?? _monotonicMicros {
    _sourceStarts.addListener(_onSourceStart);
    _models?.addListener(_onModels);
    _publish();
  }

  final CaptureMemoryShot _capture;
  final VisualMemoryEmbedder _embedder;
  final EnableMemoryImages _enableImages;
  final ValueListenable<int> _sourceStarts;
  final ValueListenable<bool> _blackFrames;
  final ValueListenable<LiveState> _liveState;

  /// A question's capture is pending in the live pipeline.
  final bool Function() _capturePending;

  /// The model rows, for the embedder's state; null in tests.
  final ValueListenable<Map<ModelId, ModelState>>? _models;

  /// Null: no timer (tests call [tick]).
  final Duration? _tickInterval;
  final MemorySampler _sampler;
  final MemoryStore _store;
  final double _countScore;
  final int _failuresUntilOff;
  final double _minCosine;
  final double _minMargin;
  final Duration _separation;
  final double _sameScene;
  final double _tie;

  /// When a frame was seen ([MemoryEntry.atMicros]) and how long ago: the
  /// wall clock, not a monotonic one: the frames outlive a trip to the
  /// background (the camera released, the device maybe asleep), and
  /// Android's monotonic clock stops while the device sleeps, which would
  /// make "seen N ago" too short. A clock set back meanwhile gives an age of
  /// zero, never a negative one ([_ageOf]).
  final int Function() _wallClock;

  /// The sampler's times ([MemorySampler]: the gap between attempts, the
  /// forced store): a monotonic clock, so a wall clock set back (a time-zone
  /// or network time change) never stops sampling, nor does one set forward
  /// make a frame due early.
  final int Function() _monotonic;

  static int _wallMicros() => DateTime.now().microsecondsSinceEpoch;

  static final Stopwatch _monotonicWatch = Stopwatch()..start();
  static int _monotonicMicros() => _monotonicWatch.elapsedMicroseconds;

  /// How long ago [atMicros] (wall clock) was, never negative.
  Duration _ageOf(int atMicros) =>
      Duration(microseconds: math.max(0, _wallClock() - atMicros));

  final ValueNotifier<VisualMemoryStatus> _status = ValueNotifier(
    const VisualMemoryStatus(),
  );

  Object? _owner;
  bool Function()? _idle;
  Timer? _timer;

  /// The camera settings' "Visual memory" switch.
  bool _enabled = true;

  /// An attempt (capture, then embedding) is in flight.
  bool _busy = false;

  /// The last tick found the demo busy: the next idle one restarts the gap.
  bool _sawBusy = false;

  /// The image encoder's load, while it runs.
  Future<void>? _widening;

  /// Why loading the image encoder failed; kept until [clear] or a new
  /// owner (a new source session does not retry the reload).
  String? _imagesFailure;

  /// Bumped by every clear: an attempt that started before is dropped.
  int _generation = 0;
  int _nextId = 0;
  int _failures = 0;
  int _failuresInARow = 0;
  String? _failureReason;
  double? _lastBestCosine;
  String? _loggedOff;
  bool _closed = false;

  /// What the memory holds and how it is doing (frames, span, embed time,
  /// failures, why it is off); changes per stored frame, clear and question.
  ValueListenable<VisualMemoryStatus> get status => _status;

  /// The frames, oldest first (tests and the overlay).
  @visibleForTesting
  List<MemoryEntry> get entries => _store.entries;

  /// Starts sampling for [owner] (the camera screen); [idle] says whether
  /// the demo is idle (no turn, no generation, no playback); [enabled] is
  /// the camera settings' switch ([setEnabled]). A take-over from another
  /// owner starts afresh. With the switch on, the image encoder's first
  /// load starts here when the embedder is ready.
  void start({
    required Object owner,
    required bool Function() idle,
    bool enabled = true,
  }) {
    if (_closed) return;
    if (!identical(_owner, owner)) {
      _imagesFailure = null;
      _clear('a new owner');
    }
    _owner = owner;
    _idle = idle;
    _enabled = enabled;
    _sawBusy = false;
    _timer?.cancel();
    if (_tickInterval case final interval?) {
      _timer = Timer.periodic(interval, (_) => unawaited(tick()));
    }
    _maybeLoadImages();
    _publish();
  }

  /// The camera settings' "Visual memory" switch. Off: sampling stops, every
  /// frame is forgotten and memory questions get [MemoryDisabled]; the
  /// image encoder is not loaded for it (one already loaded stays: the
  /// embedder's media only widen). On: sampling resumes while an owner holds
  /// the memory, after the image encoder's first load if it is not loaded.
  void setEnabled({required bool enabled}) {
    if (_closed || enabled == _enabled) return;
    _enabled = enabled;
    if (enabled) {
      _maybeLoadImages();
      _publish();
    } else {
      _clear('the visual memory was switched off');
    }
  }

  /// Stops sampling and forgets every frame, if [owner] still holds the
  /// memory (leaving the demo).
  void stop({required Object owner}) {
    if (_closed || !identical(_owner, owner)) return;
    _owner = null;
    _idle = null;
    _timer?.cancel();
    _timer = null;
    _clear('the camera demo was left');
  }

  /// "Clear memory": forgets every frame and any failure state (a failed
  /// image-encoder load is tried again).
  void clear() {
    _imagesFailure = null;
    _clear('cleared by the user');
    _maybeLoadImages();
  }

  /// One sampler step (the timer's; tests call it directly). Never throws.
  Future<void> tick() async {
    if (_closed || _owner == null || !_enabled || _busy) return;
    final off = _unavailableReason();
    // The embedder's own state has no notifier: the status follows it here.
    if (off != _status.value.unavailable ||
        _preparing != _status.value.preparing) {
      _publish();
    }
    if (off != null) return;
    if (_embedder.imageUnavailableReason != null) {
      // Nothing is sampled until the image encoder is loaded.
      _maybeLoadImages();
      return;
    }
    final now = _monotonic();
    if (!_isIdle()) {
      // Dropped: busy now, nothing queues. The gap restarts, here and at
      // the first idle tick after, so the next frame waits a full gap
      // after the demo turned idle.
      _sawBusy = true;
      _sampler.attemptEnded(now);
      return;
    }
    if (_sawBusy) {
      _sawBusy = false;
      _sampler.attemptEnded(now);
      return;
    }
    if (!_sampler.due(now)) return;
    if (_liveState.value is! LiveRunning) return;
    if (_blackFrames.value) return;
    _busy = true;
    final generation = _generation;
    try {
      await _attempt(now, _wallClock(), generation);
    } catch (e, st) {
      // _attempt reports through Results; a throw here is a bug, logged and
      // counted as a failed attempt rather than leaving the sampler wedged.
      debugPrint('[VisualMemory] sampling failed: $e\n$st');
      if (generation == _generation) _sampler.attemptEnded(_monotonic());
    } finally {
      _busy = false;
    }
  }

  /// One attempt that started at [sampledAt] (the monotonic clock, for the
  /// sampler) and [takenAt] (the wall clock, the frame's time).
  Future<void> _attempt(int sampledAt, int takenAt, int generation) async {
    final shot = await _capture(reference: _store.newest?.grey);
    if (_closed || generation != _generation) return;
    final MemoryShot value;
    switch (shot) {
      case Error(error: final MemoryFrameException error):
        // The worker detected the frame but could not make its memory
        // frame: a failure of the memory's own, counted like an embedding's.
        _countFailure('a memory frame could not be made', error);
        _sampler.attemptEnded(_monotonic());
        _publish();
        return;
      case Error(:final error):
        debugPrint('[VisualMemory] no frame this time: $error');
        _sampler.attemptEnded(_monotonic());
        return;
      case Ok(value: final v):
        value = v;
    }
    final png = value.png;
    if (png == null) {
      // The scene has not changed since the last stored frame. Once a store
      // is due anyway, that frame was seen again now: its time is refreshed
      // in place rather than a duplicate stored (and embedded). After a wall
      // clock set back this moves its time back, which is right for its age
      // (measured on that clock); which frame is newest is the store's
      // order, never this time (judgeMemory).
      if (_sampler.forced(sampledAt) && _store.refreshNewest(takenAt)) {
        _sampler.stored(sampledAt);
        debugPrint(
          '[VisualMemory] frame #${_store.newest?.frameId} seen again '
          '(${_change(value)}): time refreshed',
        );
        _publish();
      }
      _sampler.attemptEnded(_monotonic());
      return;
    }
    if (!_isIdle()) {
      // A question started while the frame came back: it goes first.
      debugPrint('[VisualMemory] frame #${value.frameId} dropped: busy');
      _sawBusy = true;
      _sampler.attemptEnded(_monotonic());
      return;
    }
    final watch = Stopwatch()..start();
    final embedded = await _embedder.embedImage(png);
    final took = watch.elapsed;
    if (_closed || generation != _generation) return;
    switch (embedded) {
      case Ok(value: final values):
        // Checked as stored (float32): a tiny vector may underflow to zeros.
        final vector = Float32List.fromList(values);
        if (vectorDefect(vector) case final defect?) {
          // Never stored: a frame no question could be compared with (and a
          // NaN score that would pass every acceptance test).
          _countFailure(
            'frame #${value.frameId} embedding is unusable',
            MemoryVectorException('the image vector has $defect'),
          );
          break;
        }
        final entry = MemoryEntry(
          id: _nextId++,
          frameId: value.frameId,
          atMicros: takenAt,
          png: png,
          width: value.width,
          height: value.height,
          unmirrored: value.unmirrored,
          counts: DetectionSummary.fromWindow([
            value.detections,
          ], minScore: _countScore).counts,
          vector: vector,
          grey: value.grey,
        );
        final dropped = _store.add(entry);
        _failuresInARow = 0;
        _sampler.stored(sampledAt);
        debugPrint(
          '[VisualMemory] frame #${value.frameId} stored '
          '${value.width}x${value.height} PNG ${png.length} B · worker '
          '${value.workerTime.inMilliseconds}ms · embed '
          '${took.inMilliseconds}ms · ${_change(value)} · '
          '${_store.length} frames ${_store.bytes} B'
          '${dropped == 0 ? '' : ' ($dropped dropped)'}',
        );
      case Error(:final error):
        _countFailure('frame #${value.frameId} embedding failed', error);
    }
    _sampler.attemptEnded(_monotonic(), embedTime: took);
    _publish();
  }

  /// The worker's scene-change numbers for the log: "diff 3.2, 1 cell".
  static String _change(MemoryShot shot) => switch (shot) {
    MemoryShot(difference: final diff?, changedCells: final cells?) =>
      'diff ${diff.toStringAsFixed(1)}, $cells cell${cells == 1 ? '' : 's'}',
    _ => 'diff –',
  };

  /// One failed frame ([what], [error]); after [_failuresUntilOff] in a row
  /// the memory is off with the last one as its reason.
  void _countFailure(String what, Object error) {
    _failures++;
    _failuresInARow++;
    debugPrint('[VisualMemory] $what ($_failuresInARow in a row): $error');
    if (_failuresInARow >= _failuresUntilOff) {
      _failureReason =
          '$_failuresInARow frames in a row failed; the last: $what ($error)';
    }
  }

  /// The newest remembered frame, for a memory question with nothing to
  /// search for ("What did you see?"): a [MemoryHit] without a cosine
  /// ([MemoryHit.latest]), nothing embedded; [MemoryEmpty], [MemoryDisabled]
  /// or [MemoryOff] as for [recall]. Never throws.
  Future<Result<MemoryRecall>> latest() async {
    if (_closed) return const Result.ok(MemoryOff('the memory was closed'));
    if (!_enabled) return const Result.ok(MemoryDisabled());
    if (_unavailableReason() case final String reason) {
      return Result.ok(MemoryOff(reason));
    }
    final newest = _store.newest;
    if (newest == null) return const Result.ok(MemoryEmpty());
    final age = _ageOf(newest.atMicros);
    debugPrint(
      '[VisualMemory] the newest frame #${newest.frameId} '
      '(${(age.inMilliseconds / 1000).toStringAsFixed(1)} s ago) of '
      '${_store.entries.length}',
    );
    return Result.ok(MemoryHit(entry: newest, cosine: null, age: age));
  }

  /// Searches the remembered frames for [phrase] (embedded raw) and judges
  /// them ([judgeMemory]). Ok with [MemoryHit] (ambiguous or not),
  /// [MemoryMiss], [MemoryEmpty], [MemoryDisabled] or [MemoryOff] (with the
  /// reason); Error when the phrase could not be embedded, its vector cannot
  /// be compared ([MemoryVectorException]: a NaN or infinite value, all
  /// zeros) or its length is not the frames'. Never throws: every stored
  /// vector passed the same check.
  Future<Result<MemoryRecall>> recall(String phrase) async {
    if (_closed) return const Result.ok(MemoryOff('the memory was closed'));
    if (!_enabled) return const Result.ok(MemoryDisabled());
    if (_unavailableReason() case final String reason) {
      return Result.ok(MemoryOff(reason));
    }
    if (_store.isEmpty) return const Result.ok(MemoryEmpty());
    final watch = Stopwatch()..start();
    final List<double> query;
    switch (await _embedder.embedText(phrase)) {
      case Ok(:final value):
        query = value;
      case Error(:final error):
        debugPrint('[VisualMemory] recall "$phrase": $error');
        return Result.error(error);
    }
    if (vectorDefect(query) case final defect?) {
      // A zero vector would throw in the cosine; a NaN one would score NaN
      // against every frame. Either is an embedder failure, never a miss.
      debugPrint('[VisualMemory] recall "$phrase": the vector has $defect');
      return Result.error(
        MemoryVectorException('the question\'s vector has $defect'),
      );
    }
    if (_closed) return const Result.ok(MemoryOff('the memory was closed'));
    final entries = _store.entries; // as it is now, after the await
    if (entries.isEmpty) return const Result.ok(MemoryEmpty());
    if (entries.any((e) => e.vector.length != query.length)) {
      // The embedder changed under the memory: nothing comparable.
      return Result.error(
        asException(
          StateError(
            'the question has ${query.length} dimensions, the remembered '
            'frames ${entries.first.vector.length}',
          ),
        ),
      );
    }
    final cosines = [
      for (final e in entries) cosineSimilarity(query, e.vector),
    ];
    // Non-null: entries is not empty.
    final verdict = judgeMemory(
      cosines: cosines,
      atMicros: [for (final e in entries) e.atMicros],
      sceneCosine: (i, best) =>
          cosineSimilarity(entries[i].vector, entries[best].vector),
      // Asked only for the few frames within the tie of the best.
      viewChanged: (i, best) => isSceneChange(
        sceneChange(
          entries[best].grey,
          entries[i].grey,
          cellThreshold: kMemoryCellChange,
          cellPixels: kMemoryCellPixels,
          maxShift: kMemoryMaxShift,
        ),
        meanThreshold: kMemorySceneChange,
        changedCells: kMemoryChangedCells,
      ),
      floor: _minCosine,
      margin: _minMargin,
      separation: _separation,
      sameScene: _sameScene,
      tie: _tie,
    )!;
    final picked = entries[verdict.best];
    final age = _ageOf(picked.atMicros);
    _lastBestCosine = verdict.cosine;
    debugPrint(
      '[VisualMemory] recall "$phrase" in ${watch.elapsedMilliseconds}ms: '
      'best ${verdict.cosine.toStringAsFixed(3)}, ${verdict.outcome.name} '
      'frame #${picked.frameId} '
      '(${cosines[verdict.best].toStringAsFixed(3)}, '
      '${(age.inMilliseconds / 1000).toStringAsFixed(1)} s ago) rival '
      '${verdict.rival?.toStringAsFixed(3) ?? '–'} over ${entries.length} '
      'frames: ${verdict.reason}',
    );
    _publish();
    switch (verdict.outcome) {
      case MemoryOutcome.accepted || MemoryOutcome.ambiguous:
        return Result.ok(
          MemoryHit(
            entry: picked,
            cosine: cosines[verdict.best],
            rival: verdict.rival,
            age: age,
            ambiguous: verdict.outcome == MemoryOutcome.ambiguous,
          ),
        );
      case MemoryOutcome.rejected:
        return Result.ok(
          MemoryMiss(
            cosine: verdict.cosine,
            rival: verdict.rival,
            // Never negative, like an age (a wall clock set back).
            span: _ageOf(entries.first.firstSeenMicros),
            frames: entries.length,
            reason: verdict.reason,
          ),
        );
    }
  }

  /// Stops sampling and releases the status. Safe to call more than once.
  void close() {
    if (_closed) return;
    _closed = true;
    _timer?.cancel();
    _timer = null;
    _owner = null;
    _idle = null;
    _sourceStarts.removeListener(_onSourceStart);
    _models?.removeListener(_onModels);
    _store.clear();
    _status.dispose();
  }

  bool _isIdle() => !_capturePending() && (_idle?.call() ?? false);

  void _onSourceStart() => _clear('a new detection session started');

  void _onModels() {
    _maybeLoadImages();
    _publish();
  }

  /// The image encoder is missing and nothing else stands in the way: an
  /// owner holds the memory with the switch on, the embedder is ready and
  /// no load failed. Its load is running or about to start.
  bool get _preparing =>
      _owner != null &&
      _enabled &&
      _imagesFailure == null &&
      _unavailableReason() == null &&
      _embedder.imageUnavailableReason != null;

  /// Starts the image encoder's load when [_preparing] says it is due and
  /// none runs. Its outcome lands in the status: ready (sampling starts at
  /// the next tick) or off with the reason.
  void _maybeLoadImages() {
    if (_closed || _widening != null || !_preparing) return;
    debugPrint("[VisualMemory] loading EmbeddingGemma's image encoder");
    final watch = Stopwatch()..start();
    late final Future<void> widening;
    widening =
        () async {
          Result<void> result;
          try {
            result = await _enableImages();
          } catch (e) {
            // A Result by contract; a throw is reported, not lost.
            result = Result.error(asException(e));
          }
          if (_closed) return;
          switch (result) {
            case Ok():
              debugPrint(
                '[VisualMemory] image encoder ready in '
                '${watch.elapsedMilliseconds}ms',
              );
            case Error(:final error):
              _imagesFailure = 'loading its image encoder failed ($error)';
              debugPrint('[VisualMemory] $_imagesFailure');
          }
        }().whenComplete(() {
          if (identical(_widening, widening)) _widening = null;
          _publish();
        });
    _widening = widening;
    // Never throws: the outcome is in the status.
    unawaited(widening);
  }

  void _clear(String why) {
    if (_closed) return;
    _generation++;
    final had = _store.length;
    _store.clear();
    _sampler.reset();
    _failures = 0;
    _failuresInARow = 0;
    _failureReason = null;
    _lastBestCosine = null;
    if (had > 0) debugPrint('[VisualMemory] $had frames forgotten: $why');
    _publish();
  }

  /// Why the memory is off; null while it works or prepares its image
  /// encoder ([_preparing]).
  String? _unavailableReason() {
    if (_imagesFailure case final String reason) return reason;
    if (_failureReason case final String reason) return reason;
    if (_models != null) {
      switch (_models.value[ModelId.embeddingGemma]) {
        case ModelFailed(:final message):
          return 'EmbeddingGemma failed to load: $message';
        case ModelUnavailable(:final reason):
          return 'EmbeddingGemma is unavailable: $reason';
        case null ||
            ModelPending() ||
            ModelInstalling() ||
            ModelLoading() ||
            ModelWarmingUp():
          return 'EmbeddingGemma is still loading';
        case ModelReady():
          break;
      }
    }
    // Images not loaded yet are a load to make (preparing), not a reason to
    // be off; an embedder that cannot load them fails that load.
    return null;
  }

  void _publish() {
    if (_closed) return;
    final unavailable = _unavailableReason();
    if (unavailable != _loggedOff) {
      _loggedOff = unavailable;
      debugPrint(
        unavailable == null
            ? '[VisualMemory] available'
            : '[VisualMemory] off: $unavailable',
      );
    }
    final oldest = _store.oldest;
    _status.value = VisualMemoryStatus(
      active: _owner != null,
      enabled: _enabled,
      preparing: _preparing,
      frames: _store.length,
      bytes: _store.bytes,
      span: oldest == null ? Duration.zero : _ageOf(oldest.firstSeenMicros),
      lastEmbed: _sampler.lastEmbed == Duration.zero
          ? null
          : _sampler.lastEmbed,
      failures: _failures,
      failuresInARow: _failuresInARow,
      unavailable: unavailable,
      lastBestCosine: _lastBestCosine,
    );
  }
}
