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
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show TextSkillExecutor;
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:provider/single_child_widget.dart';

import '../data/repositories/audio_repository.dart';
import '../data/repositories/audio_repository_device.dart';
import '../data/repositories/chat_model_repository.dart';
import '../data/repositories/conversation_repository.dart';
import '../data/repositories/conversation_repository_edge_ai.dart';
import '../data/repositories/diagnostics_repository.dart';
import '../data/repositories/hardware_repository.dart';
import '../data/repositories/image_repository.dart';
import '../data/repositories/knowledge_repository.dart';
import '../data/repositories/live_camera_settings_repository.dart';
import '../data/repositories/live_detection_repository.dart';
import '../data/repositories/model_repository.dart';
import '../data/repositories/provisioning_repository.dart';
import '../data/repositories/skill_repository.dart';
import '../data/repositories/speech_repository.dart';
import '../data/services/audio/audio_device_service.dart';
import '../data/services/audio/audio_session_service.dart';
import '../data/services/audio/mic_service.dart';
import '../data/services/audio/pcm_player_service.dart';
import '../data/services/detector/detector_service.dart';
import '../data/services/frames/camera_frame_source.dart';
import '../data/services/frames/fixture_frame_source.dart';
import '../data/services/frames/frame_source.dart';
import '../data/services/frames/network_frame_source.dart';
import '../data/services/hardware/hardware_info_service.dart';
import '../data/services/hardware/memory_probe.dart';
import '../data/services/hardware/native_log_tap.dart';
import '../data/services/images/image_input_service.dart';
import '../data/services/knowledge/embedder_digests.dart';
import '../data/services/knowledge/embedder_service.dart';
import '../data/services/knowledge/kb_documents.dart';
import '../data/services/knowledge/kb_index_marker.dart';
import '../data/services/knowledge/kb_prebuilt_index.dart';
import '../data/services/knowledge/vector_store_service.dart';
import '../data/services/llm/llm_service.dart';
import '../data/services/model_store/bundled_model_files.dart';
import '../data/services/model_store/model_file_picker.dart';
import '../data/services/model_store/model_store.dart';
import '../data/services/model_store/models_folder.dart';
import '../data/services/settings/settings_store.dart';
import '../data/services/settings/typed_settings.dart';
import '../data/services/skills/skill_store_service.dart';
import '../data/services/speech/stt_service.dart';
import '../data/services/speech/tts_service.dart';
import '../domain/models/audio_devices.dart';
import '../domain/models/frame_source_spec.dart';
import '../domain/models/hardware_profile.dart' show HostPlatform;
import '../domain/models/model_id.dart';
import '../domain/models/model_state.dart' show ModelReady;
import '../domain/ports/model_file_picker.dart';
import '../domain/ports/model_states.dart';
import '../domain/ports/self_test_launcher.dart';
import '../domain/skills/app_intent_executor.dart';
import '../domain/skills/app_intent_handlers.dart';
import '../domain/use_cases/chat_model_switcher.dart';
import '../domain/use_cases/gpu_arbiter.dart';
import '../selftest/self_test_in_app.dart';
import '../selftest/self_test_options.dart';
import '../utils/result.dart';
import 'bootstrap.dart';
import 'build_info.dart';
import 'live_camera_config.dart';
import 'model_catalog.dart' show kDefineChatModel;
import 'voice_config.dart';

/// The app's object graph: every service and repository is
/// built here, none inside the classes that use it. Services stay private;
/// repositories are exposed through [providers] — the model repository only
/// as its read-only [ModelStates] ([models]): what changes the models (setup,
/// reloads, the recognizer and detector switches) is the explicit calls
/// below, wired to the view models that need them, or goes through
/// [chatModelSwitcher]. Each route creates its own view model (and, for the
/// voice demos, its own `VoiceAssistant`).
final class AppDependencies {
  AppDependencies._({
    required this._provisioning,
    required this.chatModels,
    required this.picker,
    required this._models,
    required this.diagnostics,
    required this.hardware,
    required this.conversation,
    required this.live,
    required this._audio,
    required this._speech,
    required this._images,
    required this.knowledge,
    required this._liveSettings,
    required this._skills,
    required this.intents,
    required this.chatModelSwitcher,
    required this._nativeLog,
    required this._store,
    required this._selfTestOptions,
    required this._arbiter,
  });

  /// Initializes flutter_edge_ai and builds services, then repositories.
  /// [mic] replaces the device microphone and [imageInput] the image picker
  /// (integration tests feed fixture audio and pictures through them).
  /// [frameSource] replaces `FRAME_SOURCE`/`FIXTURE_DIR` for Demo 3.
  /// [modelStoreRoot] replaces the model store's folder, where Android also
  /// extracts the built-in models.
  /// [logTap] replaces the native log tap
  /// (release builds on Linux and macOS redirect native stderr into
  /// `<app support>/logs/native.log`).
  /// [selfTestOptions] are the in-app self-test's (a check on a device in a
  /// rack skips the audio step: no one listens to it).
  static Future<AppDependencies> create({
    MicService? mic,
    ImageInputService? imageInput,
    Result<FrameSourceSpec>? frameSource,
    Future<Directory> Function()? modelStoreRoot,
    ModelFilePicker picker = const FileSelectorModelFilePicker(),
    NativeLogTap? logTap,
    SelfTestOptions? selfTestOptions,
  }) async {
    // A bad VOICE_GATE_DBFS fails startup here, visibly, not on first use.
    final voiceConfig = kVoiceConfig;
    debugPrint(
      '[Voice] silence gate ${voiceConfig.silenceGateDbfs} dBFS, '
      '${voiceConfig.minVoiced.inMilliseconds} ms voiced',
    );
    // Before anything native loads: the GPU lines of every load land in it.
    final platform = HostPlatform.fromOperatingSystem(Platform.operatingSystem);
    final nativeLog =
        logTap ??
        nativeLogTapFor(
          path:
              '${(await getApplicationSupportDirectory()).path}/logs/native.log',
          platform: platform,
        );
    echoUnhandledErrorsToStdout(nativeLog);
    await initEdgeAi();

    final services = _createServices(modelStoreRoot: modelStoreRoot);
    final repos = await _createRepositories(
      services,
      platform: platform,
      nativeLog: nativeLog,
      mic: mic,
      imageInput: imageInput,
      frameSource: frameSource,
    );
    final models = repos.models;

    // 5. App-scoped use cases: the detector pauses while the chat model
    //    generates (the pause names it); the switcher owns the chat model's
    //    reloads.
    final arbiter = GpuArbiter(
      llmBusy: repos.conversation.isGenerating,
      setDetectorDuty: repos.live.setDuty,
      duringGeneration: kDetectorDuringGeneration,
      chatModelName: () => switch (models.states.value[ModelId.chat]) {
        ModelReady(:final info) => info.chat?.name ?? kDefineChatModel.name,
        _ => null,
      },
    );
    final chatModelSwitcher = ChatModelSwitcher(
      conversation: repos.conversation,
      reloadChatModel: models.reloadChatModel,
      unloadChatModel: models.unloadChatModel,
      refuseChatModelLoads: models.refuseChatModelLoads,
    );

    return AppDependencies._(
      provisioning: repos.provisioning,
      chatModels: repos.chatModels,
      picker: picker,
      models: models,
      diagnostics: repos.diagnostics,
      hardware: repos.hardware,
      conversation: repos.conversation,
      live: repos.live,
      audio: repos.audio,
      speech: repos.speech,
      images: repos.images,
      knowledge: repos.knowledge,
      liveSettings: repos.liveSettings,
      skills: repos.skills,
      intents: repos.intents,
      chatModelSwitcher: chatModelSwitcher,
      nativeLog: nativeLog,
      store: services.store,
      // The in-app self-test loads the detector where the app does.
      selfTestOptions:
          selfTestOptions ??
          SelfTestOptions(detectorBackend: standardDetectorBackend()),
      arbiter: arbiter,
    );
  }

  /// 1. Services.
  static _Services _createServices({
    required Future<Directory> Function()? modelStoreRoot,
  }) {
    final store = ModelStore(root: modelStoreRoot);
    return (
      store: store,
      // Android extracts the built-in models into the store's `bundled/`.
      bundled: BundledModelFiles(storeRoot: store.root),
      llm: LlmService(),
      stt: SttService(),
      tts: TtsService(),
      detector: DetectorService(),
      embedder: EmbedderService(),
    );
  }

  /// 2. Repositories, 3. the skill executors, then 4. the conversation and
  /// diagnostics, in that order.
  static Future<_Repositories> _createRepositories(
    _Services services, {
    required HostPlatform platform,
    required NativeLogTap nativeLog,
    required MicService? mic,
    required ImageInputService? imageInput,
    required Result<FrameSourceSpec>? frameSource,
  }) async {
    final (:store, :bundled, :llm, :stt, :tts, :detector, :embedder) = services;

    // 2. Repositories. The model repository loads the built-in models. The
    //    chat model slot holds the user's own .litertlm (or, with none
    //    chosen, the GEMMA_MODEL_PATH define), kept in the model store or
    //    used in place: its saved choice is read before anything can load.
    final settings = TypedSettings(store: SharedPreferencesSettingsStore());
    final chatModels = ChatModelRepository(
      settings: settings,
      store: store,
      folders: defaultModelsFolders(),
    );
    await chatModels.load();
    final provisioning = ProvisioningRepository(
      store: store,
      chatModels: chatModels,
    );
    // Demo 3's camera source and detector backend: the build's defines (or
    // the test's source) win; the detector reads its backend at every load.
    final liveSource = frameSource ?? frameSourceFromEnvironment();
    final liveSettings = LiveCameraSettingsRepository(
      settings: settings,
      environment: liveSource,
    );
    final models = ModelRepository(
      bundled: bundled,
      llm: llm,
      stt: stt,
      tts: tts,
      detector: detector,
      detectorBackendChoice: liveSettings.readBackend,
      embedder: embedder,
      logTap: nativeLog,
      chatModels: chatModels,
    );
    // The voice demos' mic and speaker. An injected mic (integration tests'
    // fixture audio) has no OS device to check.
    final audio = DeviceAudioRepository(
      session: audioSessionServiceFor(platform),
      mic: mic ?? RecordMicService(),
      player: SoloudPcmPlayerService(),
      deviceService: mic == null
          ? audioDeviceServiceFor(platform)
          : _InjectedMicDeviceService(audioDeviceServiceFor(platform)),
      platform: platform,
    );
    // Chip, GPU and RAM, probed once in the background; the "This device"
    // card and Copy diagnostics read it (and the audio devices once a voice
    // demo checked them).
    final hardware = HardwareRepository(
      service: hardwareInfoServiceForPlatform(),
      models: models.states,
      memory: memoryProbeFor(platform),
      logTap: nativeLog,
      audio: audio.devices,
      build: currentBuildInfo(),
    );
    unawaited(hardware.probe());
    // Follows the embedder's state: indexes once it is ready, never blocks
    // setup.
    final knowledge = KnowledgeRepository(
      embedder: embedder,
      store: VectorStoreService(embedQuery: embedder.embedQuery),
      models: models.states,
      digests: FileEmbedderDigests(cacheDir: defaultKbIndexDir),
      documents: AssetKbDocumentSource(),
      prebuilt: AssetPrebuiltKbIndex(),
    );
    final live = LiveDetectionRepository(
      detector: detector,
      createSource: createFrameSource,
    );
    final speech = SpeechRepository(stt: stt, tts: tts);
    final images = ImageRepository(
      input: imageInput ?? PlatformImageInputService(),
    );

    // The runtime skills, app-scoped.
    final skills = SkillRepository(store: SkillStoreService());

    // 3. Skill executors: device_info reads the hardware
    //    report when called.
    final appIntents = AppIntentExecutor(
      skillNamed: (name) {
        final loaded = skills.catalog.value?.agentSkills;
        if (loaded == null) return null;
        for (final skill in loaded) {
          if (skill.name == name) return skill;
        }
        return null;
      },
      buildAppIntents(
        deviceFacts: () => describeHardware(
          hardware: hardware.profile,
          models: hardware.modelDiagnostics(),
        ),
      ),
    );
    // 4. The conversation; it knows the LLM service and the executors, no
    //    other repository.
    final conversation = EdgeAiConversationRepository(
      llm: llm,
      executors: [appIntents, TextSkillExecutor()],
    );
    final diagnostics = DiagnosticsRepository(
      models: models.states,
      liveState: live.state,
      liveStats: live.stats,
      llmBusy: conversation.isGenerating,
      knowledge: knowledge.status,
      activeStt: models.activeStt,
      sttSwitchError: models.sttSwitchError,
      skills: skills.catalog,
      audioDevices: audio.devices,
    );
    // The first scan runs in the background; Demo 1 rescans when it opens.
    unawaited(skills.refresh());

    return (
      provisioning: provisioning,
      chatModels: chatModels,
      liveSettings: liveSettings,
      models: models,
      audio: audio,
      hardware: hardware,
      knowledge: knowledge,
      live: live,
      speech: speech,
      images: images,
      skills: skills,
      intents: appIntents,
      conversation: conversation,
      diagnostics: diagnostics,
    );
  }

  /// Releases the chat and reloads (or frees) the chat model after its
  /// choice changed.
  final ChatModelSwitcher chatModelSwitcher;

  /// The native log tap every model load reads (the self-test's too).
  final NativeLogTap _nativeLog;

  /// The model store; closed last.
  final ModelStore _store;

  /// The Models screen's "Run self-test".
  SelfTestLauncher get selfTest => _selfTest;
  late final InAppSelfTest _selfTest = InAppSelfTest(
    switcher: chatModelSwitcher,
    chatModels: chatModels,
    models: _models.states,
    logTap: _nativeLog,
    audio: _audio,
    options: _selfTestOptions,
  );
  final SelfTestOptions _selfTestOptions;

  /// The runtime skills (app-scoped).
  final SkillRepository _skills;

  /// The app intents, also run directly for the requests Demo 1
  /// recognizes itself (the time, device facts).
  final AppIntentExecutor intents;

  /// The app came back to the foreground: dropped-in skills are rescanned
  /// (Demo 1 applies a change when idle).
  void onResumed() {
    if (_disposing != null) return;
    unawaited(_skills.refresh());
  }

  /// Which models are present for loading (the Models screen).
  final ProvisioningRepository _provisioning;

  /// Which chat model the demos use; the user's own `.litertlm`.
  final ChatModelRepository chatModels;

  /// The Models screen's folder or file picker.
  final ModelFilePicker picker;

  /// The models, read-only: their states, setup running, required ready.
  ModelStates get models => _models;
  final ModelRepository _models;

  /// The calls that change the models, each wired to the view model that
  /// needs it: setup (the Models screen), a demo's recognizer on entry, and
  /// Demo 3's detector reload after a backend switch.
  Future<Result<void>> prepareModels() => _models.prepareAll();
  Future<Result<void>> activateStt(ModelId id) => _models.activateStt(id);
  Future<Result<void>> reloadDetector() => _models.reloadDetector();

  /// The active speech recognizer (integration tests).
  ValueListenable<ActiveStt?> get activeStt => _models.activeStt;

  final DiagnosticsRepository diagnostics;

  /// The device probe, the "This device" card and Copy diagnostics.
  final HardwareRepository hardware;
  final ConversationRepository conversation;
  final LiveDetectionRepository live;
  final AudioRepository _audio;
  final SpeechRepository _speech;
  final ImageRepository _images;
  final KnowledgeRepository knowledge;

  /// Demo 3's settings: the camera source (device or network camera) and
  /// where the detector runs. The build's `FRAME_SOURCE` (or the test's
  /// source) locks the source unless it is the user's choice
  /// (`FRAME_SOURCE=camera`, the default).
  final LiveCameraSettingsRepository _liveSettings;
  final GpuArbiter _arbiter;

  Future<void>? _disposing;

  List<SingleChildWidget> get providers => [
    Provider<ProvisioningRepository>.value(value: _provisioning),
    Provider<ChatModelRepository>.value(value: chatModels),
    Provider<ModelStates>.value(value: _models),
    Provider<DiagnosticsRepository>.value(value: diagnostics),
    Provider<HardwareRepository>.value(value: hardware),
    Provider<ConversationRepository>.value(value: conversation),
    Provider<LiveDetectionRepository>.value(value: live),
    Provider<AudioRepository>.value(value: _audio),
    Provider<SpeechRepository>.value(value: _speech),
    Provider<ImageRepository>.value(value: _images),
    Provider<KnowledgeRepository>.value(value: knowledge),
    Provider<SkillRepository>.value(value: _skills),
    Provider<LiveCameraSettingsRepository>.value(value: _liveSettings),
  ];

  /// Closes everything, each user before what it uses (not the creation
  /// order): a self-test in flight first (stopped, bounded by
  /// [kSelfTestStopGrace]; it holds its own engine and the app's audio),
  /// then the listeners — the arbiter (the chat), diagnostics (model, live
  /// and knowledge states), the hardware repository (model states), the
  /// skills, the knowledge base (model states) — then live detection (its
  /// frame in flight comes back first), the chat, the audio (mic, player),
  /// the models (embedder, detector worker, TTS, STT, the chat model), the
  /// chat model choice, and last the model store. Safe to call twice; the
  /// closes log native failures instead of throwing.
  Future<void> dispose() => _disposing ??= _dispose();

  Future<void> _dispose() async {
    // Its run would otherwise go on under the closes below, and load the
    // app's chat model again when it ends.
    await _selfTest.close();
    _arbiter.dispose();
    diagnostics.dispose();
    hardware.dispose();
    _skills.dispose();
    await knowledge.close();
    await live.close();
    await conversation.close();
    await _audio.close();
    await _models.close();
    chatModels.dispose();
    // Last: the chat model's download or import still running is cancelled
    // (its partial file is kept for a resume).
    await _store.close();
  }
}

/// The app's frame sources: what [LiveDetectionRepository.start] opens for
/// each [FrameSourceSpec] (a new source every time; sources are
/// single-use).
Result<FrameSource> createFrameSource(FrameSourceSpec spec) => switch (spec) {
  CameraSourceSpec() => Result.ok(CameraFrameSource()),
  FixtureSourceSpec(:final paths, :final mirrored) => Result.ok(
    FixtureFrameSource(paths: paths, mirrored: mirrored),
  ),
  NetworkSourceSpec(:final url) => Result.ok(NetworkFrameSource(url: url)),
};

/// The input check for an injected microphone (integration tests' fixture
/// audio): there is no OS device behind it, so the system's input must not
/// decide. The output still meets the real system.
final class _InjectedMicDeviceService implements AudioDeviceService {
  const _InjectedMicDeviceService(this._system);

  final AudioDeviceService _system;

  @override
  Future<DeviceCheck> checkInput() async =>
      const DeviceReady('injected microphone', detail: 'test fixture');

  @override
  Future<AudioSystem?> audioSystem() => _system.audioSystem();
}

/// What [AppDependencies._createServices] builds.
typedef _Services = ({
  ModelStore store,
  BundledModelFiles bundled,
  LlmService llm,
  SttService stt,
  TtsService tts,
  DetectorService detector,
  EmbedderService embedder,
});

/// What [AppDependencies._createRepositories] builds: the repositories, and
/// the app intents the conversation runs.
typedef _Repositories = ({
  ProvisioningRepository provisioning,
  ChatModelRepository chatModels,
  LiveCameraSettingsRepository liveSettings,
  ModelRepository models,
  DeviceAudioRepository audio,
  HardwareRepository hardware,
  KnowledgeRepository knowledge,
  LiveDetectionRepository live,
  SpeechRepository speech,
  ImageRepository images,
  SkillRepository skills,
  AppIntentExecutor intents,
  EdgeAiConversationRepository conversation,
  DiagnosticsRepository diagnostics,
});
