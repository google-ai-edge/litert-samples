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

// A port: an interface that the domain and the view models depend on and an
// outer layer implements. Every port lives in lib/domain/ports/;
// function-typed dependencies stay next to their one consumer.

import 'package:flutter/foundation.dart' show ValueListenable;

import '../models/model_id.dart';
import '../models/model_state.dart';

/// What the screens may read about the models: each one's state, whether a
/// setup run goes, and whether every required model is ready.
/// `ModelRepository` implements it; the screens get this read-only view, and
/// what changes the models (setup, reloads) is wired to them explicitly.
abstract interface class ModelStates {
  /// One entry per model; replaced (never mutated) on every change.
  ValueListenable<Map<ModelId, ModelState>> get states;

  /// True while a setup run goes.
  ValueListenable<bool> get preparing;

  /// True when every required model is [ModelReady]; optional ones may have
  /// failed or be unavailable.
  bool get requiredReady;
}
