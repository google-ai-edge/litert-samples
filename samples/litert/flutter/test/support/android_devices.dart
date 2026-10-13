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

// Android phones as the hardware probe sees them: `getprop` output in its
// dump format (`[key]: [value]`) and `/proc/meminfo`. Property values follow
// what these phones report; the RAM figures are typical for the model.

import 'package:litert_edge_demos/data/services/hardware/android_props.dart';

import 'hardware_trees.dart';

/// Galaxy S24 (US, Snapdragon 8 Gen 3 for Galaxy), Android 14, 8 GB.
const kGalaxyS24Snapdragon = '''
[ro.board.platform]: [pineapple]
[ro.build.version.release]: [14]
[ro.build.version.sdk]: [34]
[ro.hardware]: [qcom]
[ro.product.manufacturer]: [samsung]
[ro.product.model]: [SM-S921U]
[ro.soc.manufacturer]: [QTI]
[ro.soc.model]: [SM8650]
''';

/// Galaxy S24 (Europe, Exynos 2400), Android 14.
const kGalaxyS24Exynos = '''
[ro.board.platform]: [s5e9945]
[ro.build.version.release]: [14]
[ro.build.version.sdk]: [34]
[ro.hardware]: [s5e9945]
[ro.product.manufacturer]: [samsung]
[ro.product.model]: [SM-S921B]
[ro.soc.manufacturer]: [Samsung]
[ro.soc.model]: [s5e9945]
''';

/// An Android 11 phone: no `ro.soc.*`, only the board.
const kAndroid11Kona = '''
[ro.board.platform]: [kona]
[ro.build.version.release]: [11]
[ro.build.version.sdk]: [30]
[ro.hardware]: [qcom]
[ro.product.manufacturer]: [OnePlus]
[ro.product.model]: [IN2023]
[ro.soc.manufacturer]: []
[ro.soc.model]: []
''';

/// A MediaTek phone the table does not know.
const kMediatekPhone = '''
[ro.board.platform]: [mt6989]
[ro.build.version.release]: [15]
[ro.build.version.sdk]: [35]
[ro.hardware]: [mt6989]
[ro.product.manufacturer]: [Xiaomi]
[ro.product.model]: [24117RK2CG]
[ro.soc.manufacturer]: [Mediatek]
[ro.soc.model]: [MT6989]
''';

/// `/proc/meminfo` of an 8 GB phone (the kernel reserves some).
const kPhoneMeminfo =
    'MemTotal:        7620344 kB\n'
    'MemFree:          312456 kB\n'
    'MemAvailable:    2903456 kB\n'
    'Buffers:            4096 kB\n';

/// The parsed properties of [dump].
MapAndroidProperties propsOf(String dump) =>
    MapAndroidProperties(parseGetpropDump(dump));

/// A phone's `/proc` with [meminfo] and a kernel release.
FakeSystemFiles phoneFiles({String? meminfo = kPhoneMeminfo}) =>
    FakeSystemFiles({
      '/proc/meminfo': ?meminfo,
      '/proc/sys/kernel/osrelease': '6.1.75-android14-11-g1234\n',
    });
