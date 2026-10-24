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

// ignore_for_file: prefer_initializing_formals

import 'dart:convert';
import 'dart:io';

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../core/engines/engine_pool.dart';
import '../core/power/power_source.dart';
import '../core/world/time_of_day.dart';

/// Everything the demo needs to know that is not code: which model file,
/// which context window, which backends to sweep, and whether to run the
/// comparison unattended (for headless verification via `flutter run`).
///
/// All values come from `--dart-define` so the same binary serves the macOS
/// dev machine and the Windows demo laptop.
class DemoConfig {
  const DemoConfig({
    required this.modelPath,
    required this.maxTokens,
    required this.backends,
    required this.prompt,
    required this.warmup,
    required this.autorun,
    required this.autorunExit,
    required this.resultsPath,
    required this.policy,
    required this.autorunRepeat,
    required this.maxOutputTokens,
    required this.topK,
    required this.act,
    required this.act2,
    required this.act3,
    required this.powerFile,
    required this.powerMode,
    this.npuModelPath,
    this.errors = const [],
  });

  /// Compile-time `--dart-define`s, overridable at RUNTIME by a JSON file
  /// named in the `DEMO_CONFIG` environment variable (same keys). The
  /// runtime path exists so a Windows build can be reused across runs
  /// without rebuilding for every parameter change.
  ///
  /// Never throws: every unreadable file or bad value lands in [errors]
  /// (and the key keeps its default), so `main` can report them all at once
  /// instead of the app sitting on the home screen with a silently ignored
  /// config.
  factory DemoConfig.fromEnvironment() => DemoConfig.fromOverrides(_runtimeOverrides());

  /// [runtime] is the parsed `DEMO_CONFIG` object and the error reading it
  /// produced, if any. Separate from [DemoConfig.fromEnvironment] for tests.
  factory DemoConfig.fromOverrides((Map<String, Object?>, String?) runtime) {
    final k = _Keys(runtime);

    final modelPath = k.str('MODEL_PATH', const String.fromEnvironment('MODEL_PATH'));
    final maxTokens = k.integer('MAX_TOKENS', const int.fromEnvironment('MAX_TOKENS', defaultValue: 4096));
    final backends = k.str('BACKENDS', const String.fromEnvironment('BACKENDS', defaultValue: 'cpu,gpu,npu'));
    final prompt = k.str('PROMPT', const String.fromEnvironment('PROMPT', defaultValue: defaultPrompt));
    final promptFile = k.str('PROMPT_FILE', const String.fromEnvironment('PROMPT_FILE'));
    final warmup = k.boolean('WARMUP', const bool.fromEnvironment('WARMUP', defaultValue: true));
    final autorun = k.boolean('AUTORUN', const bool.fromEnvironment('AUTORUN'));
    final autorunExit = k.boolean('AUTORUN_EXIT', const bool.fromEnvironment('AUTORUN_EXIT'));
    final resultsPath = k.str('RESULTS_PATH', const String.fromEnvironment('RESULTS_PATH'));
    final policy = k.str('POLICY', const String.fromEnvironment('POLICY', defaultValue: 'resident'));
    final autorunRepeat = k.integer('AUTORUN_REPEAT', const int.fromEnvironment('AUTORUN_REPEAT', defaultValue: 1));
    final maxOutputTokens = k.integer('MAX_OUTPUT_TOKENS', const int.fromEnvironment('MAX_OUTPUT_TOKENS', defaultValue: 0));
    final topK = k.integer('TOP_K', const int.fromEnvironment('TOP_K', defaultValue: 1));
    final act = k.integer('ACT', const int.fromEnvironment('ACT', defaultValue: 0));
    final act2 = Act2Spec(
      modelPath: k.str('ACT2_MODEL_PATH', const String.fromEnvironment('ACT2_MODEL_PATH')),
      backend: k.backend('ACT2_BACKEND', k.str('ACT2_BACKEND', const String.fromEnvironment('ACT2_BACKEND', defaultValue: 'gpu'))),
      maxTokens: k.integer('ACT2_MAX_TOKENS', const int.fromEnvironment('ACT2_MAX_TOKENS', defaultValue: 16384)),
      contrastMaxTokens: k.integer('ACT2_CONTRAST_MAX_TOKENS', const int.fromEnvironment('ACT2_CONTRAST_MAX_TOKENS', defaultValue: 4096)),
      contrastModelPath: k.str('ACT2_CONTRAST_MODEL_PATH', const String.fromEnvironment('ACT2_CONTRAST_MODEL_PATH')),
      transcriptPath: k.str('ACT2_TRANSCRIPT', const String.fromEnvironment('ACT2_TRANSCRIPT')),
      autorunQuestion: k.str('ACT2_QUESTION', const String.fromEnvironment('ACT2_QUESTION')),
      maxOutputTokens: k.integer('ACT2_MAX_OUTPUT_TOKENS', const int.fromEnvironment('ACT2_MAX_OUTPUT_TOKENS', defaultValue: 300)),
    );
    // An NPU build has one fixed window, so the long-context model and the
    // small contrast model cannot both default to NPU_MODEL_PATH.
    if (act2.backend == PreferredBackend.npu && act2.modelPathOr('').isEmpty) {
      k.errors.add('ACT2_MODEL_PATH: required when ACT2_BACKEND is npu (an NPU build of the long-context model)');
    }

    // ACT3_NPU_MODEL_PATH is the older name, from when only Act 3 had it.
    var npuModel = k.str('NPU_MODEL_PATH', const String.fromEnvironment('NPU_MODEL_PATH'));
    if (npuModel.isEmpty) npuModel = k.str('ACT3_NPU_MODEL_PATH', const String.fromEnvironment('ACT3_NPU_MODEL_PATH'));
    if (npuModel.isNotEmpty && !File(npuModel).existsSync()) k.errors.add('NPU_MODEL_PATH: no file at $npuModel');
    final recordPath = k.str('ACT3_RECORD', const String.fromEnvironment('ACT3_RECORD'));
    final dayStart = k.str('ACT3_TIME_OF_DAY', const String.fromEnvironment('ACT3_TIME_OF_DAY', defaultValue: 'day'));
    final act3 = Act3Spec(
      npuModelPath: npuModel.isNotEmpty ? npuModel : null,
      narratorMaxTokens: k.integer('ACT3_NARRATOR_MAX_TOKENS', const int.fromEnvironment('ACT3_NARRATOR_MAX_TOKENS', defaultValue: 4096)),
      narratorIntervalMs: k.integer('ACT3_NARRATOR_INTERVAL_MS', const int.fromEnvironment('ACT3_NARRATOR_INTERVAL_MS', defaultValue: 4000)),
      narratorPlan: k.plan('ACT3_NARRATOR_PLAN', const String.fromEnvironment('ACT3_NARRATOR_PLAN')),
      scene: k.str('ACT3_SCENE', const String.fromEnvironment('ACT3_SCENE', defaultValue: 'runner')),
      viewChunks: k.integer('ACT3_VIEW_CHUNKS', const int.fromEnvironment('ACT3_VIEW_CHUNKS', defaultValue: 7)),
      propDensity: k.number('ACT3_PROP_DENSITY', const String.fromEnvironment('ACT3_PROP_DENSITY', defaultValue: '1.0')),
      shadows: k.boolean('ACT3_SHADOWS', const bool.fromEnvironment('ACT3_SHADOWS', defaultValue: true)),
      post: k.boolean('ACT3_POST', const bool.fromEnvironment('ACT3_POST', defaultValue: true)),
      calibrate: k.boolean('ACT3_CALIBRATE', const bool.fromEnvironment('ACT3_CALIBRATE', defaultValue: true)),
      targetFps: k.integer('ACT3_TARGET_FPS', const int.fromEnvironment('ACT3_TARGET_FPS', defaultValue: 60)),
      baseline: k.boolean('ACT3_BASELINE', const bool.fromEnvironment('ACT3_BASELINE', defaultValue: true)),
      pilotMissRate: k.number('ACT3_PILOT_MISS_RATE', const String.fromEnvironment('ACT3_PILOT_MISS_RATE', defaultValue: '0.08')),
      bodyCount: k.integer('ACT3_BODIES', const int.fromEnvironment('ACT3_BODIES', defaultValue: 60)),
      renderScale: k.number('ACT3_RENDER_SCALE', const String.fromEnvironment('ACT3_RENDER_SCALE', defaultValue: '1.0')),
      autorunSeconds: k.integer('ACT3_AUTORUN_SECONDS', const int.fromEnvironment('ACT3_AUTORUN_SECONDS', defaultValue: 0)),
      dayCycleSeconds: k.number('ACT3_DAY_SECONDS', const String.fromEnvironment('ACT3_DAY_SECONDS', defaultValue: '240')),
      recordPath: recordPath.isEmpty ? null : recordPath,
      recordFps: k.integer('ACT3_RECORD_FPS', const int.fromEnvironment('ACT3_RECORD_FPS', defaultValue: 30)),
      recordScale: k.number('ACT3_RECORD_SCALE', const String.fromEnvironment('ACT3_RECORD_SCALE', defaultValue: '1.0')),
      startZ: k.number('ACT3_START_Z', const String.fromEnvironment('ACT3_START_Z', defaultValue: '4')),
      details: k.boolean('ACT3_DETAILS', const bool.fromEnvironment('ACT3_DETAILS', defaultValue: true)),
      dayStart: DayPhase.values.asNameMap()[dayStart] ??
          k.fail('ACT3_TIME_OF_DAY', dayStart, 'expected ${DayPhase.values.map((p) => p.name).join('|')}', DayPhase.day),
    );

    final powerFile = k.str('POWER_FILE', const String.fromEnvironment('POWER_FILE'));
    final powerModeRaw = k.str('POWER_MODE', const String.fromEnvironment('POWER_MODE', defaultValue: 'auto'));

    var promptText = prompt;
    if (promptFile.isNotEmpty) {
      try {
        promptText = File(promptFile).readAsStringSync();
      } catch (e) {
        k.errors.add('PROMPT_FILE: cannot read $promptFile ($e)');
      }
    }
    if (modelPath.isNotEmpty && !File(modelPath).existsSync()) {
      k.errors.add('MODEL_PATH: no file at $modelPath');
    }

    return DemoConfig(
      npuModelPath: npuModel.isNotEmpty ? npuModel : null,
      powerFile: powerFile.isNotEmpty ? powerFile : null,
      powerMode: PowerMode.values.asNameMap()[powerModeRaw] ??
          k.fail('POWER_MODE', powerModeRaw, 'expected auto|watts|utilization', PowerMode.auto),
      modelPath: modelPath,
      maxTokens: maxTokens,
      backends: [
        for (final b in backends.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty))
          k.backend('BACKENDS', b),
      ],
      prompt: promptText,
      warmup: warmup,
      autorun: autorun,
      autorunExit: autorunExit,
      resultsPath: resultsPath.isNotEmpty ? resultsPath : null,
      autorunRepeat: autorunRepeat,
      maxOutputTokens: maxOutputTokens > 0 ? maxOutputTokens : null,
      topK: topK,
      act: act,
      act2: act2,
      act3: act3,
      policy: switch (policy) {
        'resident' => EnginePolicy.resident,
        'sequential' => EnginePolicy.sequential,
        _ => k.fail('POLICY', policy, 'expected resident|sequential', EnginePolicy.resident),
      },
      errors: List.unmodifiable(k.errors),
    );
  }

  /// Problems found while reading the configuration, one line each. Empty
  /// when every key was understood.
  final List<String> errors;

  /// A run nobody is watching: it must exit on a bad config, not wait.
  bool get headless => autorunExit;

  /// Absolute path to the `.litertlm` file (`MODEL_PATH`), read in place.
  /// Empty when unset: the scene still runs, every model row says so.
  final String modelPath;

  /// The model for the NPU in every act (`NPU_MODEL_PATH`): an NPU-compiled
  /// build. The standard file does not load on the NPU (the engine has no
  /// fallback), so without this the NPU row fails. Null = [modelPath].
  final String? npuModelPath;

  String modelPathFor(PreferredBackend b) =>
      b == PreferredBackend.npu && npuModelPath != null ? npuModelPath! : modelPath;

  /// Context window for every backend. NOTE: on NPU this must equal the
  /// `cache_length` the bundle was compiled for (not clamped there).
  final int maxTokens;

  /// Backends to sweep, in order. Runs are always sequential.
  final List<PreferredBackend> backends;

  /// The measured prompt. Long enough that prefill is visible.
  final String prompt;

  /// Run a short unmeasured prompt after each engine load so shader/kernel
  /// warm-up does not land in the measured TTFT.
  final bool warmup;

  /// Start the comparison as soon as the app is up (headless verification).
  final bool autorun;

  /// After an autorun finishes, print the JSON summary and exit the process.
  final bool autorunExit;

  /// When set, the final comparison is written there as JSON. Independent of
  /// stdout, which the plugin redirects to a file in debug builds.
  final String? resultsPath;

  /// `resident` keeps every backend loaded (instant switch, more RAM);
  /// `sequential` reloads on each switch.
  final EnginePolicy policy;

  /// How many full sweeps an autorun performs (2+ shows resident vs reload).
  final int autorunRepeat;

  /// Cap on generated tokens per measured run (null = model decides).
  final int? maxOutputTokens;

  /// Sampler top-k for measured runs (1 = greedy).
  final int topK;

  /// Act to open directly (0 = home screen).
  final int act;

  /// Act 2 (long context) settings — its own model, backend and window.
  final Act2Spec act2;

  /// Act 3 (scene) load dials and headless run length.
  final Act3Spec act3;

  /// JSON written by `tools/power_sidecar/power_sidecar.ps1`; null = platform default.
  final String? powerFile;

  /// auto | watts | utilization (see [PowerMode]).
  final PowerMode powerMode;

  String get modelFileName => fileName(modelPath);

  /// Last path segment, whichever separator the path was written with.
  static String fileName(String path) => path.split(RegExp(r'[\\/]')).last;

  static const defaultPrompt =
      'You are a technical writer. In about 150 words, explain to a laptop '
      'buyer what a neural processing unit (NPU) is, how it differs from a '
      'GPU, and why running a language model on it can save battery. '
      'Use plain language and no bullet points.';

  static (Map<String, Object?>, String?) _runtimeOverrides() {
    final path = Platform.environment['DEMO_CONFIG'];
    if (path == null || path.isEmpty) return (const {}, null);
    try {
      final decoded = jsonDecode(File(path).readAsStringSync());
      if (decoded is Map<String, Object?>) return (decoded, null);
      return (const {}, 'DEMO_CONFIG: $path is not a JSON object');
    } catch (e) {
      return (const {}, 'DEMO_CONFIG: cannot read $path ($e)');
    }
  }

  static PreferredBackend parseBackend(String s) => _parseBackend(s);

  static PreferredBackend _parseBackend(String s) => switch (s.toLowerCase()) {
        'cpu' => PreferredBackend.cpu,
        'gpu' => PreferredBackend.gpu,
        'npu' => PreferredBackend.npu,
        _ => throw ArgumentError.value(s, 'BACKENDS', 'expected cpu|gpu|npu'),
      };
}

/// Act 2 runs a different (larger) model than Acts 1/3 and never on NPU.
class Act2Spec {
  const Act2Spec({
    required String modelPath,
    required this.backend,
    required this.maxTokens,
    required this.contrastMaxTokens,
    required String contrastModelPath,
    required String transcriptPath,
    required String autorunQuestion,
    required this.maxOutputTokens,
  })  : _modelPath = modelPath,
        _contrastModelPath = contrastModelPath,
        _transcriptPath = transcriptPath,
        _autorunQuestion = autorunQuestion;
  // (private fields keep the "empty means unset" rule out of call sites)

  final String _modelPath;
  final String _contrastModelPath;
  final String _transcriptPath;
  final String _autorunQuestion;

  final PreferredBackend backend;

  /// Context window of the main (long-context) engine.
  final int maxTokens;

  /// Context window of the contrast engine that the source must NOT fit into.
  final int contrastMaxTokens;

  final int maxOutputTokens;

  /// Falls back to the Act 1/3 model when unset (e.g. before the 12B bundle
  /// is available on this machine).
  String modelPathOr(String fallback) => _modelPath.isNotEmpty ? _modelPath : fallback;

  /// The contrast engine is a small model, not a second copy of the main
  /// one: a second 12B engine would load its ~6 GB of weights again just to
  /// fail. Falls back to the Act 1/3 model (E2B).
  String contrastModelPathOr(String fallback) => _contrastModelPath.isNotEmpty ? _contrastModelPath : fallback;

  /// `[mm:ss] Speaker: text` lines. Null → bundled sample transcript.
  String? get transcriptPath => _transcriptPath.isNotEmpty ? _transcriptPath : null;

  /// When set, the act asks this question as soon as prefill is done
  /// (headless verification).
  String? get autorunQuestion => _autorunQuestion.isNotEmpty ? _autorunQuestion : null;
}

class Act3Spec {
  const Act3Spec({
    required this.bodyCount,
    required this.renderScale,
    required this.autorunSeconds,
    required this.narratorMaxTokens,
    required this.narratorIntervalMs,
    required this.narratorPlan,
    required this.npuModelPath,
    required this.scene,
    required this.viewChunks,
    required this.propDensity,
    required this.shadows,
    required this.post,
    required this.pilotMissRate,
    required this.calibrate,
    required this.targetFps,
    required this.baseline,
    required this.dayCycleSeconds,
    required this.dayStart,
    required this.details,
    required this.startZ,
    required this.recordPath,
    required this.recordFps,
    required this.recordScale,
  });

  /// Record the act to this .mp4 (frames read back in-app, piped to ffmpeg).
  /// For showing, not measuring: the read-back costs GPU time.
  final String? recordPath;
  final int recordFps;

  /// Video pixels per logical pixel (`ACT3_RECORD_SCALE`, default 1.0). On a
  /// Retina Mac 2.0 keeps the frame as drawn: `DEMO_WINDOW=1280x720` → 1440p.
  final double recordScale;

  /// Where along the trail Dash starts, metres (default 4). For captures:
  /// put a river or an obstacle a few seconds ahead instead of half a
  /// minute of running away.
  final double startZ;

  /// Start with the detail panels (frame stats, dials) shown; `H` toggles.
  /// Off gives the clean stage view: meters and commentary only.
  final bool details;

  /// Length of the runner's day-night cycle in seconds; 0 freezes the light
  /// at [dayStart] (e.g. for A/B frame-cost runs per phase).
  final double dayCycleSeconds;

  /// Part of the day the run starts in (`ACT3_TIME_OF_DAY`: morning, day,
  /// sunset, night, dawn).
  final DayPhase dayStart;

  /// Put the scene on the edge of the frame budget at startup. Without this
  /// the act can show nothing: a scene with headroom loses no frames when the
  /// model takes the GPU.
  final bool calibrate;

  /// Frame budget the calibrator aims at (`ACT3_TARGET_FPS`, default 60;
  /// 0 = the display's refresh rate). A display that cannot show clearly
  /// more than this makes its refresh rate the budget (see `FrameBudget`).
  final int targetFps;

  /// Measure each backend's solo tokens/sec (scene at its lightest) before
  /// the act, so "model speed" is a share of its own speed and not of its
  /// best reading under load.
  final bool baseline;

  /// `runner` (Dash on a generated trail) or `arena` (the original crate box,
  /// kept as a fallback until the runner is verified on the Windows box).
  final String scene;
  bool get isRunner => scene != 'arena';

  /// Runner dials — the GPU load of the act.
  final int viewChunks;
  final double propDensity;
  final bool shadows;
  final bool post;

  /// How often the autopilot fumbles an obstacle on purpose (0..1). A perfect
  /// run has nothing for the narrator to talk about.
  final double pilotMissRate;

  /// NPU-compiled bundle for the narrator's NPU row (an NPU build of Gemma 4
  /// E2B on Lunar Lake). Null = same file as the other backends.
  final String? npuModelPath;

  final int bodyCount;
  final double renderScale;

  /// Narrator context window (the event log accumulates in it).
  final int narratorMaxTokens;

  /// How often a batch of events becomes a commentary turn.
  final int narratorIntervalMs;

  /// Headless phases `backend:seconds` (`off`, `cpu`, `gpu`, `npu`), e.g.
  /// `off:10,gpu:20,npu:20`. Empty = interactive.
  final List<(String, int)> narratorPlan;

  /// > 0: run the scene this long, print frame stats, exit (headless).
  final int autorunSeconds;
}

/// Reads keys from the `DEMO_CONFIG` JSON, falling back to the compile-time
/// value, and collects every problem instead of guessing. JSON numbers,
/// booleans and their string spellings ("3", "true") are all accepted.
class _Keys {
  _Keys((Map<String, Object?>, String?) runtime) : o = runtime.$1 {
    if (runtime.$2 != null) errors.add(runtime.$2!);
  }

  final Map<String, Object?> o;
  final List<String> errors = [];

  T fail<T>(String key, Object? value, String expected, T fallback) {
    errors.add('$key: $expected, got ${jsonEncode(value)}');
    return fallback;
  }

  String str(String key, String dflt) {
    final v = o[key];
    if (v == null) return dflt;
    if (v is String) return v;
    if (v is num || v is bool) return '$v';
    return fail(key, v, 'expected a string', dflt);
  }

  int integer(String key, int dflt) {
    final v = o[key];
    if (v == null) return dflt;
    if (v is int) return v;
    if (v is double && v == v.roundToDouble()) return v.toInt();
    if (v is String && int.tryParse(v.trim()) != null) return int.parse(v.trim());
    return fail(key, v, 'expected a whole number', dflt);
  }

  /// [compileTime] is the dart-define's string (dart-defines have no double).
  double number(String key, String compileTime) {
    final v = o[key] ?? compileTime;
    if (v is num) return v.toDouble();
    if (v is String && double.tryParse(v.trim()) != null) return double.parse(v.trim());
    return fail(key, v, 'expected a number', 0);
  }

  bool boolean(String key, bool dflt) {
    final v = o[key];
    if (v == null) return dflt;
    if (v is bool) return v;
    if (v == 'true') return true;
    if (v == 'false') return false;
    return fail(key, v, 'expected true or false', dflt);
  }

  PreferredBackend backend(String key, String value) => switch (value.trim().toLowerCase()) {
        'cpu' => PreferredBackend.cpu,
        'gpu' => PreferredBackend.gpu,
        'npu' => PreferredBackend.npu,
        _ => fail(key, value, 'expected cpu|gpu|npu', PreferredBackend.cpu),
      };

  /// `gpu:30,cpu:30`: phase name (off|cpu|gpu|npu) and seconds.
  List<(String, int)> plan(String key, String compileTime) {
    final raw = str(key, compileTime);
    final out = <(String, int)>[];
    for (final part in raw.split(',').map((x) => x.trim()).where((x) => x.isNotEmpty)) {
      final bits = part.split(':');
      final name = bits.first.trim().toLowerCase();
      final secs = bits.length == 2 ? int.tryParse(bits[1].trim()) : null;
      if (!const {'off', 'cpu', 'gpu', 'npu'}.contains(name) || secs == null || secs <= 0) {
        fail(key, part, 'expected phase:seconds with phase off|cpu|gpu|npu', null);
        continue;
      }
      out.add((name, secs));
    }
    return out;
  }
}
