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

import 'package:flutter/foundation.dart';

/// Keys for tests.
abstract final class DebugOverlayKeys {
  static const toggle = ValueKey('debug-overlay-toggle');
  static const panel = ValueKey('debug-overlay-panel');

  /// The panel's scroll view (the full list scrolls inside the panel).
  static const scroll = ValueKey('debug-overlay-scroll');

  /// The handle that collapses and expands the panel.
  static const expand = ValueKey('debug-overlay-expand');

  /// The scroll strip on the panel's right edge, while its text overflows.
  static const scrollbar = ValueKey('debug-overlay-scrollbar');
}
