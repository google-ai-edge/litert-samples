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

import '../../../domain/models/model_id.dart';
import '../../../domain/models/model_state.dart';
import '../../../utils/result.dart';
import '../../services/hardware/native_log_tap.dart';

/// A step of a model load.
enum LoadStep { install, load, warmUp }

/// How a model load ended.
sealed class const LoadOutcome();

/// Loaded and warmed up; [info] is what its ready row shows.
final class const LoadReady(final LoadedModelInfo info) extends LoadOutcome;

/// [step] failed with [error]; no later step ran.
final class const LoadFailed(final LoadStep step, final Exception error)
    extends LoadOutcome;

/// close() ran between two steps (or during the last one): no later step
/// ran, and what loaded was released.
final class const LoadStopped() extends LoadOutcome;

/// The steps every model's load shares: install with progress, load, warm up.
/// It publishes [ModelInstalling] (then each new percent), [ModelLoading] and
/// [ModelWarmingUp]; the final state is the caller's, because what a failure
/// means differs per model (a required model stops setup, the detector offers
/// the other backend, missing embedder files are "unavailable").
final class ModelLoadPipeline {
  ModelLoadPipeline({required this._publish, required this._isClosed});

  final void Function(ModelId id, ModelState state) _publish;

  /// close() ran: checked after the install, the load and the warm-up.
  final bool Function() _isClosed;

  /// Runs [id]'s steps in order and stops at the first failure:
  ///
  /// - [install] registers the files, reporting 0–100 to its callback; none
  ///   for a model loaded straight from its file (the detector).
  /// - [load] builds the runtime; [releaseOrphan] closes what it loaded when
  ///   close() ran during the load or the warm-up (the model has no owner
  ///   then).
  /// - [warmUp] runs once so the first real use is not the slow one; none
  ///   for a model whose load already ran it (the detector). When it fails,
  ///   [releaseFailed] unloads what [load] loaded and keeps the service
  ///   usable, so a broken model does not hold its memory (GBs for a chat
  ///   model) until a Retry; none for a service whose warm-up does that
  ///   itself (the embedder).
  /// - [logTap]: this load's window of the native log, from before [load]
  ///   through the last step, goes to [describe]; none, an empty one.
  /// - [describe] makes the ready row from [load]'s value and [warmUp]'s
  ///   time (zero without that step).
  Future<LoadOutcome> run<T>(
    ModelId id, {
    Future<Result<Object?>> Function(void Function(int percent) onProgress)?
    install,
    required Future<Result<T>> Function() load,
    Future<void> Function()? releaseOrphan,
    Future<Result<Duration>> Function()? warmUp,
    Future<void> Function()? releaseFailed,
    NativeLogTap? logTap,
    required LoadedModelInfo Function(
      T loaded,
      Duration warmUpTime,
      List<String> nativeLog,
    )
    describe,
  }) async {
    if (install != null) {
      _publish(id, const ModelInstalling());
      int? lastPercent;
      final installed = await install((percent) {
        if (percent == lastPercent) return;
        lastPercent = percent;
        _publish(id, ModelInstalling(percent: percent));
      });
      if (installed case Error(:final error)) {
        return LoadFailed(LoadStep.install, error);
      }
      if (_isClosed()) return const LoadStopped();
    }

    _publish(id, const ModelLoading());
    final mark = logTap?.mark();
    final T loaded;
    switch (await load()) {
      case Ok(:final value):
        loaded = value;
      case Error(:final error):
        return LoadFailed(LoadStep.load, error);
    }
    if (_isClosed()) {
      await releaseOrphan?.call();
      return const LoadStopped();
    }

    var warmUpTime = Duration.zero;
    if (warmUp != null) {
      _publish(id, const ModelWarmingUp());
      final warmedUp = await warmUp();
      if (_isClosed()) {
        await releaseOrphan?.call();
        return const LoadStopped();
      }
      switch (warmedUp) {
        case Ok(:final value):
          warmUpTime = value;
        case Error(:final error):
          await releaseFailed?.call();
          return LoadFailed(LoadStep.warmUp, error);
      }
    }
    final nativeLog = logTap == null || mark == null
        ? const <String>[]
        : logTap.since(mark);
    return LoadReady(describe(loaded, warmUpTime, nativeLog));
  }
}
