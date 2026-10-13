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
import 'package:litert_edge_demos/domain/hardware/soc_table.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

void main() {
  group('Qualcomm parts (the brief\'s table)', () {
    const expected = {
      'SM8450': ('Snapdragon 8 Gen 1', 'Adreno 730', 'Hexagon V69'),
      'SM8475': ('Snapdragon 8+ Gen 1', 'Adreno 730', 'Hexagon V69'),
      'SM8550': ('Snapdragon 8 Gen 2', 'Adreno 740', 'Hexagon V73'),
      'SM8650': ('Snapdragon 8 Gen 3', 'Adreno 750', 'Hexagon V75'),
      'SM8750': ('Snapdragon 8 Elite', 'Adreno 830', 'Hexagon V79'),
      'SM8850': ('Snapdragon 8 Elite Gen 5', 'Adreno 840', 'Hexagon V81'),
    };
    for (final MapEntry(key: code, value: (name, gpu, npu))
        in expected.entries) {
      test('$code → $name / $gpu / $npu', () {
        final spec = lookupSoc(code)!;
        expect(spec.vendor, SocVendor.qualcomm);
        expect(spec.name, name);
        expect(spec.gpu, gpu);
        expect(spec.npu, npu);
      });
    }

    test('the manufacturer prefix, a bin suffix and case do not matter', () {
      expect(lookupSoc('QTI SM8650')?.name, 'Snapdragon 8 Gen 3');
      expect(lookupSoc('SM8650-AC')?.name, 'Snapdragon 8 Gen 3');
      expect(lookupSoc('sm8750')?.name, 'Snapdragon 8 Elite');
      expect(qualcommCode('QTI sm8550'), 'SM8550');
    });

    test('an unknown part is null, not a guess', () {
      expect(lookupSoc('SM9999'), isNull);
      expect(lookupSoc('MT6989'), isNull);
      expect(lookupSoc(''), isNull);
    });
  });

  test('Exynos by its s5e code', () {
    expect(lookupSoc('s5e9945')?.name, 'Exynos 2400');
    expect(lookupSoc('s5e9945')?.gpu, 'Xclipse 940');
    expect(lookupSoc('S5E9925')?.name, 'Exynos 2200');
    expect(lookupSoc('s5e9925')?.gpu, 'Xclipse 920');
    expect(lookupSoc('s5e9955')?.gpu, 'Xclipse 950');
    expect(lookupSoc('s5e9945')?.vendor, SocVendor.samsung);
  });

  test('Tensor by name or board code', () {
    expect(lookupSoc('Tensor')?.gpu, 'Mali-G78 MP20');
    expect(lookupSoc('GS201')?.name, 'Google Tensor G2');
    expect(lookupSoc('Tensor G3')?.gpu, 'Mali-G715');
    expect(lookupSoc('zumapro')?.name, 'Google Tensor G4');
    expect(lookupSoc('Tensor G5')?.gpu, 'PowerVR DXT-48-1536');
    expect(lookupSoc('Tensor G3')?.vendor, SocVendor.google);
  });

  test('boards: the unambiguous Qualcomm ones map to their part; taro does '
      'not (SM8450 and SM8475 both use it)', () {
    expect(lookupBoard('pineapple')?.code, 'SM8650');
    expect(lookupBoard('kalama')?.spec.name, 'Snapdragon 8 Gen 2');
    expect(lookupBoard('sun')?.spec.gpu, 'Adreno 830');
    expect(lookupBoard('s5e9945')?.spec.name, 'Exynos 2400');
    expect(lookupBoard('zuma')?.spec.name, 'Google Tensor G3');
    expect(lookupBoard('taro'), isNull);
    expect(lookupBoard('mt6989'), isNull);
  });

  group('SocInfo.label', () {
    test('marketing name with the part', () {
      const soc = SocInfo(
        manufacturer: 'QTI',
        model: 'SM8650',
        source: 'ro.soc.model',
        name: 'Snapdragon 8 Gen 3',
      );
      expect(soc.label, 'Snapdragon 8 Gen 3 (SM8650)');
    });

    test('a name that already holds the part is said once', () {
      const soc = SocInfo(
        manufacturer: 'Google',
        model: 'Tensor G3',
        source: 'ro.soc.model',
        name: 'Google Tensor G3',
      );
      expect(soc.label, 'Google Tensor G3');
    });

    test('unknown to the table: as Android names it', () {
      const soc = SocInfo(
        manufacturer: 'Mediatek',
        model: 'MT6989',
        source: 'ro.soc.model',
      );
      expect(soc.label, 'Mediatek MT6989');
    });
  });
}
