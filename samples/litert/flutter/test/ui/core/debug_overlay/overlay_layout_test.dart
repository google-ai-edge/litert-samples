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

// The debug panel's layout policy without widgets: compact or wide from the
// window, width and height bound from the safe area above the keyboard. The
// widget-level checks (the panel on real screens, taps through it) are in
// debug_overlay_layout_test.dart.
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay/overlay_layout.dart';

void main() {
  group('compact', () {
    bool compact(Size window) =>
        DebugOverlayLayout(window: window, safe: window).compact;

    test('below 600 wide or 480 high', () {
      expect(compact(const Size(360, 780)), isTrue, reason: 'phone');
      expect(compact(const Size(780, 360)), isTrue, reason: 'landscape');
      expect(compact(const Size(599.9, 800)), isTrue);
      expect(compact(const Size(1000, 479.9)), isTrue);
    });

    test('from 600 wide and 480 high', () {
      expect(compact(const Size(600, 480)), isFalse);
      expect(compact(const Size(1280, 800)), isFalse, reason: 'desktop');
    });

    test('follows the window, not the safe area', () {
      final layout = DebugOverlayLayout(
        window: const Size(800, 600),
        safe: const Size(400, 300),
      );
      expect(layout.compact, isFalse);
    });
  });

  group('width', () {
    test('the safe width less 8 on each side, at most 440', () {
      double width(double safe) => DebugOverlayLayout(
        window: const Size(1280, 800),
        safe: Size(safe, 800),
      ).width;

      expect(width(360), 344);
      expect(width(456), 440);
      expect(width(1280), kDebugOverlayMaxWidth);
    });

    test('never negative', () {
      final layout = DebugOverlayLayout(
        window: const Size(10, 600),
        safe: const Size(10, 600),
      );
      expect(layout.width, 0);
    });
  });

  group('height bound', () {
    test('phone: 45% of the safe height', () {
      // 360×780 with a 24 status bar and a 16 home indicator.
      final layout = DebugOverlayLayout(
        window: const Size(360, 780),
        safe: const Size(360, 740),
      );
      expect(layout.compact, isTrue);
      expect(layout.width, 344);
      expect(layout.maxHeight, moreOrLessEquals(0.45 * 740));
    });

    test('desktop: 75% of the height, unless that leaves less than 160 '
        'under the panel', () {
      final layout = DebugOverlayLayout(
        window: const Size(1280, 800),
        safe: const Size(1280, 800),
      );
      expect(layout.compact, isFalse);
      expect(layout.width, 440);
      // 75% would be 600; 800 - 60 (under the app bar) - 160 = 580.
      expect(layout.maxHeight, 580);
    });

    test('landscape phone: 160 kept free under the panel', () {
      // 780×360 with 47 side insets and a 21 home indicator.
      final layout = DebugOverlayLayout(
        window: const Size(780, 360),
        safe: const Size(686, 339),
      );
      expect(layout.compact, isTrue);
      expect(layout.width, 440);
      // 45% would be 152.55; 339 - 60 - 160 = 119.
      expect(layout.maxHeight, 119);
    });

    test('with the keyboard open: from the room above it', () {
      // A phone with a 24 status bar and a 300 keyboard.
      final layout = DebugOverlayLayout(
        window: const Size(360, 780),
        safe: const Size(360, 456),
      );
      expect(layout.maxHeight, moreOrLessEquals(0.45 * 456));
    });

    test('at least 72, when the window has room for it', () {
      final layout = DebugOverlayLayout(
        window: const Size(800, 200),
        safe: const Size(800, 200),
      );
      // The share leaves nothing (200 - 60 - 160 < 0); 200 - 68 = 132 fit.
      expect(layout.maxHeight, 72);
    });

    test('never past 8 above the safe area bottom', () {
      final layout = DebugOverlayLayout(
        window: const Size(800, 124),
        safe: const Size(800, 124),
      );
      expect(layout.maxHeight, 124 - 60 - 8);
    });

    test('never negative', () {
      final layout = DebugOverlayLayout(
        window: const Size(800, 40),
        safe: const Size(800, 40),
      );
      expect(layout.maxHeight, 0);
    });
  });

  group('showsPanel', () {
    bool shows(Size safe) =>
        DebugOverlayLayout(window: safe, safe: safe).showsPanel;

    test('room for a line and the handle: 56 high and 160 wide', () {
      expect(shows(const Size(800, 124)), isTrue, reason: '56 high');
      expect(shows(const Size(176, 600)), isTrue, reason: '160 wide');
      expect(shows(const Size(360, 740)), isTrue);
    });

    test('no panel in a window too short or too narrow', () {
      expect(shows(const Size(800, 120)), isFalse, reason: '52 high');
      expect(shows(const Size(800, 80)), isFalse);
      expect(shows(const Size(150, 600)), isFalse, reason: '134 wide');
      expect(shows(const Size(175, 600)), isFalse, reason: '159 wide');
    });
  });

  test('the panel hangs 60 under the safe top (the app bar and 4) and 8 '
      'from its right edge, exactly its width wide', () {
    expect(DebugOverlayLayout.margin, const EdgeInsets.only(top: 60, right: 8));
    final layout = DebugOverlayLayout(
      window: const Size(360, 780),
      safe: const Size(360, 740),
    );
    expect(
      layout.constraints,
      BoxConstraints(minWidth: 344, maxWidth: 344, maxHeight: layout.maxHeight),
    );
  });
}
