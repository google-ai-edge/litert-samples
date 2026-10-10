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

/// What the app knows about the phone SoCs Android names: the marketing name,
/// the GPU and the NPU. The GPU and NPU are never queried, so everything from
/// this table is *inferred*.
library;

/// Who designed a SoC.
enum SocVendor { qualcomm, samsung, google }

/// One known SoC.
final class const SocSpec({
  required final SocVendor vendor,

  /// `Snapdragon 8 Gen 3`, `Exynos 2400`, `Google Tensor G3`.
  required final String name,

  /// `Adreno 750`; null when the app does not know it reliably.
  final String? gpu,

  /// `Hexagon V75`, `Samsung NPU`, `Google TPU`.
  final String? npu,
});

/// Qualcomm parts by `ro.soc.model` (`SM8650`; a `-AC` suffix is ignored).
/// Hexagon versions match flutter_litert's `libQnnHtpV*` (V69–V81). The
/// SM8850's Adreno 840 is from Qualcomm's launch coverage (Sept 2025), not
/// from a device.
const _qualcomm = {
  // SM8250 is the 865, the 865+ and the 870 (SM8250-AC): one name for all.
  'SM8250': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 865/865+/870',
    gpu: 'Adreno 650',
    npu: 'Hexagon V66',
  ),
  'SM8350': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 888',
    gpu: 'Adreno 660',
    npu: 'Hexagon V68',
  ),
  'SM8450': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 8 Gen 1',
    gpu: 'Adreno 730',
    npu: 'Hexagon V69',
  ),
  'SM8475': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 8+ Gen 1',
    gpu: 'Adreno 730',
    npu: 'Hexagon V69',
  ),
  'SM8550': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 8 Gen 2',
    gpu: 'Adreno 740',
    npu: 'Hexagon V73',
  ),
  'SM8635': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 8s Gen 3',
    gpu: 'Adreno 735',
    npu: 'Hexagon',
  ),
  'SM8650': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 8 Gen 3',
    gpu: 'Adreno 750',
    npu: 'Hexagon V75',
  ),
  'SM7675': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 7+ Gen 3',
    gpu: 'Adreno 732',
    npu: 'Hexagon',
  ),
  'SM8750': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 8 Elite',
    gpu: 'Adreno 830',
    npu: 'Hexagon V79',
  ),
  'SM8850': SocSpec(
    vendor: SocVendor.qualcomm,
    name: 'Snapdragon 8 Elite Gen 5',
    gpu: 'Adreno 840',
    npu: 'Hexagon V81',
  ),
};

/// Samsung Exynos and Google Tensor by `ro.soc.model` or board name,
/// lower-case. Tensor G3 and G4 both carry a Mali-G715 (sources disagree on
/// the core count, so none is given).
const _others = {
  'exynos2100': SocSpec(
    vendor: SocVendor.samsung,
    name: 'Exynos 2100',
    gpu: 'Mali-G78 MP14',
    npu: 'Samsung NPU',
  ),
  's5e9925': SocSpec(
    vendor: SocVendor.samsung,
    name: 'Exynos 2200',
    gpu: 'Xclipse 920',
    npu: 'Samsung NPU',
  ),
  's5e9945': SocSpec(
    vendor: SocVendor.samsung,
    name: 'Exynos 2400',
    gpu: 'Xclipse 940',
    npu: 'Samsung NPU',
  ),
  's5e9955': SocSpec(
    vendor: SocVendor.samsung,
    name: 'Exynos 2500',
    gpu: 'Xclipse 950',
    npu: 'Samsung NPU',
  ),
  'tensor': _tensor,
  'gs101': _tensor,
  'tensor g2': _tensorG2,
  'gs201': _tensorG2,
  'tensor g3': _tensorG3,
  'zuma': _tensorG3,
  'tensor g4': _tensorG4,
  'zumapro': _tensorG4,
  'tensor g5': SocSpec(
    vendor: SocVendor.google,
    name: 'Google Tensor G5',
    gpu: 'PowerVR DXT-48-1536',
    npu: 'Google TPU',
  ),
};

const _tensor = SocSpec(
  vendor: SocVendor.google,
  name: 'Google Tensor',
  gpu: 'Mali-G78 MP20',
  npu: 'Google TPU',
);
const _tensorG2 = SocSpec(
  vendor: SocVendor.google,
  name: 'Google Tensor G2',
  gpu: 'Mali-G710 MP7',
  npu: 'Google TPU',
);
const _tensorG3 = SocSpec(
  vendor: SocVendor.google,
  name: 'Google Tensor G3',
  gpu: 'Mali-G715',
  npu: 'Google TPU',
);
const _tensorG4 = SocSpec(
  vendor: SocVendor.google,
  name: 'Google Tensor G4',
  gpu: 'Mali-G715',
  npu: 'Google TPU',
);

/// Qualcomm board names (`ro.board.platform`) whose part is unambiguous, for
/// Android 11, which has no `ro.soc.model`. Left out: `taro` (SM8450 and
/// SM8475) and `lahaina` (SM8350, but 778G/780G phones report it too). A
/// chip found this way is labelled inferred (`SocInfo.inferred`).
const _qualcommBoards = {
  'kona': 'SM8250',
  'kalama': 'SM8550',
  'pineapple': 'SM8650',
  'sun': 'SM8750',
};

final _qualcommCode = RegExp(r'SM\d{4}', caseSensitive: false);
final _exynosCode = RegExp(r's5e\d{4}', caseSensitive: false);

/// The Qualcomm part code in [model] (`QTI SM8650` → `SM8650`), or null.
String? qualcommCode(String model) =>
    _qualcommCode.firstMatch(model)?.group(0)?.toUpperCase();

/// The table's entry for a SoC model as Android names it (`SM8650`,
/// `QTI SM8650`, `s5e9945`, `Tensor G3`, `GS201`); null when unknown.
SocSpec? lookupSoc(String model) {
  if (qualcommCode(model) case final code?) return _qualcomm[code];
  if (_exynosCode.firstMatch(model)?.group(0) case final code?) {
    return _others[code.toLowerCase()];
  }
  return _others[model.trim().toLowerCase()];
}

/// The table's entry for a board name (`pineapple`, `s5e9945`, `zuma`), and
/// the part code it stands for; null when unknown or ambiguous.
({String code, SocSpec spec})? lookupBoard(String board) {
  final key = board.trim().toLowerCase();
  if (_qualcommBoards[key] case final code?) {
    return (code: code, spec: _qualcomm[code]!);
  }
  if (_others[key] case final spec?) return (code: key, spec: spec);
  return null;
}
