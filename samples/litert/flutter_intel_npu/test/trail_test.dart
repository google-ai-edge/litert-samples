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
import 'package:intel_npu_demo/core/world/trail.dart';

void main() {
  test('toTrail with the right hint lands on the branch the point came from', () {
    // The mesher, Dash and the props all ask "where on the trail is this
    // point" with the trail z they started from. Near the tight bend's fold
    // the iteration used to jump to the other branch, tens of metres away.
    const trail = Trail();
    var jumps = 0;
    var worst = 0.0;
    for (var z = 0.0; z <= 3000; z += 1) {
      for (var lateral = -46.0; lateral <= 46; lateral += 2) {
        final p = trail.pointAt(z, lateral: lateral);
        final (zt, _) = trail.toTrail(p.x, p.y, guess: z);
        final off = (zt - z).abs();
        if (off > 0.5) jumps++;
        if (off > worst) worst = off;
      }
    }
    expect(jumps, 0, reason: 'worst offset ${worst.toStringAsFixed(2)} m');
  });
}
