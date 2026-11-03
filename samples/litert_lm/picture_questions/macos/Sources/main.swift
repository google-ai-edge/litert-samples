// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

// One binary, two modes: the window, and the headless engine process that the window starts once per model file
// (EngineHost.swift says why).
if CommandLine.arguments.contains(EngineHost.flag) {
  EngineHost.main(arguments: CommandLine.arguments)
} else {
  // Writing to an engine that has already ended must not end the app.
  signal(SIGPIPE, SIG_IGN)
  MainActor.assumeIsolated {
    PictureQuestionsApp.main()
  }
}
