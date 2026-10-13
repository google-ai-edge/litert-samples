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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/model/model_load_pipeline.dart';
import 'package:litert_edge_demos/data/services/hardware/native_log_tap.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// The steps every model's load shares: what each step publishes, where it
/// stops, and what the ready row is made from.
void main() {
  late List<String> published;
  late List<String> calls;
  late bool closed;
  late _Log log;
  late ModelLoadPipeline pipeline;

  setUp(() {
    published = [];
    calls = [];
    closed = false;
    log = _Log();
    pipeline = ModelLoadPipeline(
      publish: (id, state) => published.add('${id.name} ${_show(state)}'),
      isClosed: () => closed,
    );
  });

  Future<Result<String>> install(void Function(int percent) onProgress) async {
    calls.add('install');
    log.lines.add('install');
    for (final percent in [10, 10, 60, 100]) {
      onProgress(percent);
    }
    return const Result.ok('model-id');
  }

  Future<Result<String>> load() async {
    calls.add('load');
    log.lines.add('load');
    return const Result.ok('loaded');
  }

  Future<Result<Duration>> warmUp() async {
    calls.add('warm-up');
    log.lines.add('warm-up');
    return const Result.ok(Duration(milliseconds: 7));
  }

  LoadedModelInfo describe(
    String loaded,
    Duration warmUpTime,
    List<String> nativeLog,
  ) => LoadedModelInfo(
    modelId: loaded,
    backend: 'cpu',
    loadTime: const Duration(milliseconds: 3),
    warmUpTime: warmUpTime,
    nativeLog: nativeLog,
  );

  test('every step in order: installing (each new percent once), loading, '
      'warming up; the ready row from the load and the warm-up', () async {
    final outcome = await pipeline.run(
      ModelId.whisperBase,
      install: install,
      load: load,
      warmUp: warmUp,
      logTap: log,
      describe: describe,
    );

    expect(published, [
      'whisperBase installing',
      'whisperBase installing 10%',
      'whisperBase installing 60%',
      'whisperBase installing 100%',
      'whisperBase loading',
      'whisperBase warming up',
    ]);
    expect(calls, ['install', 'load', 'warm-up']);
    final info = (outcome as LoadReady).info;
    expect(info.modelId, 'loaded');
    expect(info.warmUpTime, const Duration(milliseconds: 7));
    expect(info.nativeLog, [
      'load',
      'warm-up',
    ], reason: 'from before the load through the warm-up');
  });

  test('no install and no warm-up step (the detector): loading only, a zero '
      'warm-up time, the log window ends with the load', () async {
    log.lines.add('before');

    final outcome = await pipeline.run(
      ModelId.yolo26n,
      load: () async {
        log.lines.add('load');
        return const Result.ok('loaded');
      },
      logTap: log,
      describe: describe,
    );

    expect(published, ['yolo26n loading']);
    final info = (outcome as LoadReady).info;
    expect(info.warmUpTime, Duration.zero);
    expect(info.nativeLog, ['load']);
  });

  test('without a log tap the window is empty', () async {
    final outcome = await pipeline.run(
      ModelId.inflectNano,
      install: install,
      load: load,
      warmUp: warmUp,
      describe: describe,
    );

    expect((outcome as LoadReady).info.nativeLog, isEmpty);
  });

  test('an install failure stops before the load', () async {
    final error = Exception('the disk is full');

    final outcome = await pipeline.run(
      ModelId.chat,
      install: (onProgress) async => Result<String>.error(error),
      load: load,
      warmUp: warmUp,
      describe: describe,
    );

    expect(outcome, isA<LoadFailed>());
    expect((outcome as LoadFailed).step, LoadStep.install);
    expect(outcome.error, same(error));
    expect(published, ['chat installing']);
    expect(calls, isEmpty);
  });

  test('a load failure stops before the warm-up', () async {
    final error = Exception('wrong SoC');

    final outcome = await pipeline.run(
      ModelId.chat,
      install: install,
      load: () async => Result<String>.error(error),
      warmUp: warmUp,
      describe: describe,
    );

    expect((outcome as LoadFailed).step, LoadStep.load);
    expect(outcome.error, same(error));
    expect(published.last, 'chat loading');
    expect(calls, ['install']);
  });

  test('a warm-up failure releases what loaded (the service stays usable '
      'for a Retry)', () async {
    final error = Exception('sampler crashed');

    final outcome = await pipeline.run(
      ModelId.chat,
      install: install,
      load: load,
      releaseOrphan: () async => calls.add('release orphan'),
      warmUp: () async {
        calls.add('warm-up');
        return Result<Duration>.error(error);
      },
      releaseFailed: () async => calls.add('release failed'),
      describe: describe,
    );

    expect((outcome as LoadFailed).step, LoadStep.warmUp);
    expect(outcome.error, same(error));
    expect(published.last, 'chat warming up');
    expect(calls, ['install', 'load', 'warm-up', 'release failed']);
  });

  test('close() during the warm-up: what loaded is released, stopped (no '
      'ready row)', () async {
    final outcome = await pipeline.run(
      ModelId.chat,
      install: install,
      load: load,
      releaseOrphan: () async => calls.add('release orphan'),
      warmUp: () async {
        calls.add('warm-up');
        closed = true;
        return const Result.ok(Duration(milliseconds: 7));
      },
      releaseFailed: () async => calls.add('release failed'),
      describe: describe,
    );

    expect(outcome, isA<LoadStopped>());
    expect(calls, ['install', 'load', 'warm-up', 'release orphan']);
    expect(published.last, 'chat warming up');
  });

  test('close() during a warm-up that failed: stopped, released once as '
      'an orphan', () async {
    final outcome = await pipeline.run(
      ModelId.inflectNano,
      install: install,
      load: load,
      releaseOrphan: () async => calls.add('release orphan'),
      warmUp: () async {
        closed = true;
        return Result<Duration>.error(Exception('synth boom'));
      },
      releaseFailed: () async => calls.add('release failed'),
      describe: describe,
    );

    expect(outcome, isA<LoadStopped>());
    expect(calls, ['install', 'load', 'release orphan']);
  });

  test('close() during the install: stopped before the load', () async {
    final outcome = await pipeline.run(
      ModelId.embeddingGemma,
      install: (onProgress) async {
        closed = true;
        return const Result.ok('model-id');
      },
      load: load,
      warmUp: warmUp,
      describe: describe,
    );

    expect(outcome, isA<LoadStopped>());
    expect(calls, isEmpty);
    expect(published, ['embeddingGemma installing']);
  });

  test(
    'close() during the load: what loaded is released, no warm-up',
    () async {
      final outcome = await pipeline.run(
        ModelId.yolo26n,
        load: () async {
          closed = true;
          return const Result.ok('loaded');
        },
        releaseOrphan: () async => calls.add('release'),
        warmUp: warmUp,
        describe: describe,
      );

      expect(outcome, isA<LoadStopped>());
      expect(calls, ['release']);
      expect(published, ['yolo26n loading']);
    },
  );

  test(
    'close() during a load that failed: the failure, nothing to release',
    () async {
      final error = Exception('partly on the GPU');

      final outcome = await pipeline.run(
        ModelId.yolo26n,
        load: () async {
          closed = true;
          return Result<String>.error(error);
        },
        releaseOrphan: () async => calls.add('release'),
        describe: describe,
      );

      expect((outcome as LoadFailed).error, same(error));
      expect(calls, isEmpty);
    },
  );
}

String _show(ModelState state) => switch (state) {
  ModelInstalling(:final percent) =>
    percent == null ? 'installing' : 'installing $percent%',
  ModelLoading() => 'loading',
  ModelWarmingUp() => 'warming up',
  _ => 'unexpected $state',
};

/// A native log the steps write into; a mark is a line count.
final class _Log implements NativeLogTap {
  final List<String> lines = [];

  @override
  String get description => 'test log';

  @override
  int mark() => lines.length;

  @override
  List<String> since(int mark) => List.unmodifiable(lines.sublist(mark));
}
