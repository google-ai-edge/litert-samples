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
import 'dart:ui' show AppExitResponse;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:provider/single_child_widget.dart';

import 'config/demos.dart';
import 'config/dependencies.dart';
import 'config/model_catalog.dart';
import 'config/voice_config.dart';
import 'data/repositories/chat_model_repository.dart';
import 'data/repositories/diagnostics_repository.dart';
import 'data/repositories/hardware_repository.dart';
import 'data/repositories/knowledge_repository.dart';
import 'data/repositories/live_camera_settings_repository.dart';
import 'data/repositories/live_detection_repository.dart';
import 'data/repositories/skill_repository.dart';
import 'domain/models/camera_side_event.dart';
import 'domain/models/chat_side_event.dart';
import 'domain/ports/model_states.dart';
import 'domain/use_cases/camera_turn_responder.dart';
import 'domain/use_cases/chat_turn_responder.dart';
import 'domain/use_cases/voice_assistant.dart';
import 'ui/core/app_foreground.dart';
import 'ui/core/debug_overlay.dart';
import 'ui/features/home/view_models/home_view_model.dart';
import 'ui/features/home/views/home_screen.dart';
import 'ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'ui/features/live_camera/views/live_camera_screen.dart';
import 'ui/features/setup/view_models/chat_model_view_model.dart';
import 'ui/features/setup/view_models/self_test_view_model.dart';
import 'ui/features/setup/view_models/setup_view_model.dart';
import 'ui/features/setup/views/setup_screen.dart';
import 'ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'ui/features/voice_chat/views/voice_chat_screen.dart';

/// `/` setup → `/home` (replaces setup) → push `/chat` (Demo 1) or
/// `/camera` (Demo 3), or `/models` from home's menu. Back pops to home,
/// which disposes that route's view model.
abstract final class Routes {
  static const setup = '/';
  static const home = '/home';
  static const chat = '/chat';
  static const camera = '/camera';
  static const models = '/models';

  static String of(Demo demo) => switch (demo) {
    Demo.voiceChat => chat,
    Demo.liveCamera => camera,
  };
}

/// The Models screen's view models: the models and their setup, the chat
/// model card and the self-test.
List<SingleChildWidget> _modelsScreen(AppDependencies deps, SetupMode mode) => [
  ChangeNotifierProvider(
    create: (c) => SetupViewModel(
      models: c.read(),
      prepareModels: deps.prepareModels,
      provisioning: c.read(),
      mode: mode,
      hardware: c.read<HardwareRepository>(),
      chatModels: c.read<ChatModelRepository>(),
      switcher: deps.chatModelSwitcher,
    ),
  ),
  ChangeNotifierProvider(
    create: (c) => ChatModelViewModel(
      chatModels: c.read(),
      models: c.read(),
      switcher: deps.chatModelSwitcher,
      picker: deps.picker,
    ),
  ),
  ChangeNotifierProvider(
    create: (c) => SelfTestViewModel(
      launcher: deps.selfTest,
      blockers: [
        c.read<ModelStates>().preparing,
        c.read<ChatModelRepository>().busy,
        deps.chatModelSwitcher.busy,
      ],
    ),
  ),
];

/// Root widget. Owns [dependencies] and closes them when the app exits.
class App extends StatefulWidget {
  const App({super.key, required this.dependencies});

  final AppDependencies dependencies;

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  late final AppLifecycleListener _lifecycle;

  /// Whether the demos may hold the camera and the mic (Android and iOS:
  /// only in the foreground); the demos' view models react to it.
  final AppForeground _foreground = AppForeground();

  @override
  void initState() {
    super.initState();
    // Desktop quit: close the chat and the native model before the process
    // goes, instead of leaving the engine to process teardown. Resume:
    // rescan the skills folder. Every state: AppForeground, which the demos
    // follow to release the camera and the mic on Android and iOS.
    _lifecycle = AppLifecycleListener(
      onExitRequested: _onExitRequested,
      onResume: widget.dependencies.onResumed,
      onStateChange: _foreground.onStateChange,
    );
  }

  Future<AppExitResponse> _onExitRequested() async {
    await widget.dependencies.dispose();
    return AppExitResponse.exit;
  }

  @override
  void dispose() {
    _lifecycle.dispose();
    _foreground.dispose();
    // Unawaited on purpose: State.dispose is synchronous. AppDependencies
    // .dispose() is idempotent, and each close it runs catches and logs its
    // own native failure (EdgeAiConversationRepository.close,
    // LlmService.close), so this future does not complete with an error.
    unawaited(widget.dependencies.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final deps = widget.dependencies;
    return MultiProvider(
      providers: deps.providers,
      child: MaterialApp(
        title: 'LiteRT Demos',
        // The debug overlay already marks debug builds; the banner would hide
        // the New conversation button.
        debugShowCheckedModeBanner: false,
        theme: ThemeData(colorSchemeSeed: Colors.teal),
        darkTheme: ThemeData(
          colorSchemeSeed: Colors.teal,
          brightness: Brightness.dark,
        ),
        initialRoute: Routes.setup,
        // Dependency injection: config/dependencies.dart builds every
        // service and repository (no class builds its own collaborators or
        // reads the environment for them), and a route gets them two ways.
        // What the provider tree holds (the repositories, the read-only
        // ModelStates) is read with c.read(); what it does not hold comes
        // from deps: the app-scoped objects (the chat model switcher, the
        // picker, the self-test, the app intents) and the explicit calls that
        // change the models (prepareModels, activateStt, reloadDetector).
        routes: {
          Routes.setup: (context) => MultiProvider(
            providers: _modelsScreen(deps, SetupMode.firstRun),
            child: SetupScreen(
              onReady: () =>
                  Navigator.of(context).pushReplacementNamed(Routes.home),
            ),
          ),
          Routes.models: (context) => MultiProvider(
            providers: _modelsScreen(deps, SetupMode.manage),
            child: const SetupScreen(),
          ),
          Routes.home: (context) => ChangeNotifierProvider(
            create: (c) => HomeViewModel(
              models: c.read<ModelStates>().states,
              knowledge: c.read<KnowledgeRepository>().status,
              hardware: c.read<HardwareRepository>(),
            ),
            child: HomeScreen(
              onOpen: (demo) =>
                  Navigator.of(context).pushNamed<void>(Routes.of(demo)),
              onOpenModels: () =>
                  Navigator.of(context).pushNamed<void>(Routes.models),
            ),
          ),
          Routes.chat: (context) => ChangeNotifierProvider(
            // The view model owns this demo's assistant and disposes it.
            create: (c) => VoiceChatViewModel(
              conversation: c.read(),
              diagnostics: c.read(),
              images: c.read(),
              activateStt: deps.activateStt,
              skills: c.read<SkillRepository>(),
              foreground: _foreground,
              assistant: VoiceAssistant<ChatSideEvent>(
                speech: c.read(),
                audio: c.read(),
                responders: ChatTurnResponder(
                  conversation: c.read(),
                  retriever: c.read<KnowledgeRepository>(),
                  direct: deps.intents.run,
                ),
                diagnostics: c.read<DiagnosticsRepository>(),
                config: kVoiceConfig.withMaxUtterance(
                  kSttConfigs[Demo.voiceChat.stt]!.window,
                ),
                recognizer: Demo.voiceChat.stt,
              ),
            ),
            child: const VoiceChatScreen(),
          ),
          Routes.camera: (context) => ChangeNotifierProvider(
            // The view model owns this demo's assistant and disposes it.
            create: (c) {
              final live = c.read<LiveDetectionRepository>();
              return LiveCameraViewModel(
                conversation: c.read(),
                live: live,
                diagnostics: c.read(),
                activateStt: deps.activateStt,
                // The camera source and detector settings (the build's
                // FRAME_SOURCE or the test's source still wins).
                settings: c.read<LiveCameraSettingsRepository>(),
                models: c.read<ModelStates>().states,
                foreground: _foreground,
                reloadDetector: deps.reloadDetector,
                assistant: VoiceAssistant<CameraSideEvent>(
                  speech: c.read(),
                  audio: c.read(),
                  responders: CameraTurnResponder(
                    capture: live.capture,
                    conversation: c.read(),
                    encode: live.encodeForLlm,
                  ),
                  diagnostics: c.read<DiagnosticsRepository>(),
                  // moonshine reads 5 s: a longer press is ended there.
                  config: kVoiceConfig.withMaxUtterance(
                    kSttConfigs[Demo.liveCamera.stt]!.window,
                  ),
                  recognizer: Demo.liveCamera.stt,
                ),
              );
            },
            child: const LiveCameraScreen(),
          ),
        },
        builder: (context, child) => DebugOverlayHost(
          snapshot: deps.diagnostics.snapshot,
          child: child ?? const SizedBox.shrink(),
        ),
      ),
    );
  }
}
