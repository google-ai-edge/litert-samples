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
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:path_provider/path_provider.dart';

import '../config/build_info.dart';
import '../config/env.dart';
import '../data/repositories/chat_model_repository.dart';
import '../data/services/hardware/hardware_info_service.dart';
import '../data/services/hardware/memory_probe.dart';
import '../data/services/hardware/native_log_tap.dart';
import '../data/services/model_store/model_store.dart';
import '../data/services/model_store/models_folder.dart';
import '../data/services/settings/settings_store.dart';
import '../data/services/settings/typed_settings.dart';
import '../domain/models/chat_model.dart';
import '../domain/models/hardware_profile.dart' show HostPlatform;
import '../utils/result.dart';
import 'self_test_adapters.dart';
import 'self_test_options.dart';
import 'self_test_report.dart';
import 'self_test_runner.dart';

/// `--selftest` / `SELFTEST=1` (main.dart): runs the steps without the app's
/// UI, prints the SELFTEST block to stdout, writes it to the report file and
/// exits 0 (all passed) or 1. A small window lists the progress; nothing of
/// the normal app starts.
Future<Never> runSelfTestMode(Result<SelfTestOptions> parsed) async {
  WidgetsFlutterBinding.ensureInitialized();
  final progress = ValueNotifier<List<String>>(const []);
  runApp(_SelfTestApp(lines: progress));
  void say(String line) {
    stdout.writeln('[SELFTEST] $line');
    progress.value = [...progress.value, line];
  }

  final SelfTestOptions options;
  switch (parsed) {
    case Ok(:final value):
      options = value;
    case Error(:final error):
      say('$error');
      await stdout.flush();
      exit(1);
  }
  // Dart's own report of an uncaught error goes to fd 2, which the tap
  // redirects: say it on stdout too, and fail the run.
  final unhandled = <String>[];
  final previousOnError = PlatformDispatcher.instance.onError;
  PlatformDispatcher.instance.onError = (error, stack) {
    unhandled.add('$error');
    say('unhandled error: $error\n$stack');
    return previousOnError?.call(error, stack) ?? false;
  };

  // The watchdog and the normal path may both want to end the run: the
  // first one writes the block and exits, the other never completes.
  var ending = false;
  Future<Never> endOnce(Future<Never> Function() end) {
    if (ending) return Completer<Never>().future;
    ending = true;
    return end();
  }

  SelfTestRunner? runner;
  String? outPath;
  // A native call that never returns must not hang a headless run: past the
  // limit, report what finished and exit 1. (The models are not closed: the
  // call holding one may never return; exit() releases them.)
  final watchdog = Timer(options.timeout, () {
    say('timed out after ${options.timeout.inSeconds} s');
    final partial = runner?.report(timedOut: options.timeout);
    unawaited(
      endOnce(
        () => partial == null
            ? _crash('timed out before the steps started', say)
            : _finish(partial, outPath, say, progress),
      ),
    );
  });
  try {
    final platform = HostPlatform.fromOperatingSystem(Platform.operatingSystem);
    final stamp = DateTime.now()
        .toUtc()
        .toIso8601String()
        .split('.')
        .first
        .replaceAll(':', '-');
    final out = outPath =
        options.outPath ??
        '${(await getApplicationSupportDirectory()).path}/selftest/'
            'selftest-$stamp.txt';
    final logTap = nativeLogTapFor(
      path: '${out.replaceFirst(RegExp(r'\.txt$'), '')}.native.log',
      platform: platform,
    );
    say('native log: ${logTap.description}');

    // Gemma only: the self-test needs no speech, embedding or vector store.
    await FlutterEdgeAi.initialize(inferenceEngines: const [LiteRtLmEngine()]);

    final chatModel = await resolveSelfTestChatModel(
      argument: options.gemmaPath,
      define: kGemmaModelPath,
      plan: await _chosenChatModel(say),
    );
    final detectorFile = await resolveSelfTestDetector(
      argument: options.detectorPath,
    );
    final imagePath = options.imagePath;

    final steps = runner = SelfTestRunner(
      options: options,
      build: currentBuildInfo(),
      hardware: hardwareInfoServiceForPlatform(),
      detector: AppSelfTestDetector(),
      gemma: AppSelfTestGemma(),
      memory: memoryProbeFor(platform),
      logTap: logTap,
      chatModel: chatModel,
      detectorFile: detectorFile,
      loadImage: () => loadSelfTestImage(imagePath),
      imageLabel: imagePath ?? kSelfTestCatsLabel,
      audio: options.skipAudio ? null : AppSelfTestAudio(platform),
      progress: say,
      unhandledErrors: unhandled,
    );
    final report = await steps.run();
    watchdog.cancel();
    return await endOnce(() => _finish(report, out, say, progress));
  } catch (e, st) {
    watchdog.cancel();
    return endOnce(() => _crash('$e\n$st', say));
  }
}

/// Writes the block to [outPath] (when known), prints it, exits with the
/// report's code.
Future<Never> _finish(
  SelfTestReport report,
  String? outPath,
  void Function(String) say,
  ValueNotifier<List<String>> progress,
) async {
  String? written;
  if (outPath != null) {
    try {
      final file = File(outPath);
      await file.parent.create(recursive: true);
      await file.writeAsString(formatSelfTest(report, reportPath: outPath));
      written = outPath;
    } on FileSystemException catch (e) {
      say('could not write the report: $e');
    }
  }
  final text = formatSelfTest(report, reportPath: written);
  stdout.write(text);
  progress.value = [...progress.value, '', ...text.trimRight().split('\n')];
  await stdout.flush();
  exit(report.exitCode);
}

/// Something outside the steps failed (a bug or a broken install): still one
/// delimited FAIL block and exit 1, never a window left hanging.
Future<Never> _crash(String why, void Function(String) say) async {
  say('crashed: $why');
  stdout.write(
    '$kSelfTestBegin\nresult     FAIL · crashed outside the steps · exit 1\n'
    '${why.split('\n').first}\n$kSelfTestEnd\n',
  );
  await stdout.flush();
  exit(1);
}

/// The chat model chosen on the Models screen (the user's own `.litertlm`,
/// in place or from the store's `custom/` folder).
Future<ChatModelPlan> _chosenChatModel(void Function(String) say) async {
  final store = ModelStore();
  ChatModelRepository? chatModels;
  try {
    // Reads only: a retired saved choice is not migrated from here.
    chatModels = ChatModelRepository(
      settings: TypedSettings(store: SharedPreferencesSettingsStore()),
      store: store,
      persistMigration: false,
      folders: defaultModelsFolders(),
    );
    if (await chatModels.load() case Error(:final error)) {
      say('chat model choice: $error');
    }
    return chatModels.plan;
  } catch (e) {
    say('chat model choice unavailable: $e');
    return chatModels?.plan ?? const NoChatModelPlan();
  } finally {
    chatModels?.dispose();
    await store.close();
  }
}

/// The self-test's window: the progress lines, then the block.
class _SelfTestApp extends StatelessWidget {
  const _SelfTestApp({required this.lines});

  final ValueListenable<List<String>> lines;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(title: const Text('Self-test (see the terminal)')),
        body: ValueListenableBuilder(
          valueListenable: lines,
          builder: (context, value, _) => ListView(
            padding: const EdgeInsets.all(12),
            children: [
              for (final line in value)
                Text(
                  line,
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
