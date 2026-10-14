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

import 'dart:convert';
import 'dart:typed_data';

import 'package:litert_edge_demos/domain/models/model_towers.dart'
    show LitertlmSection;

/// One section of a synthetic `.litertlm` header: its `model_type` and its
/// `backend_constraint`, each left out when null.
typedef HeaderSection = ({String? type, String? constraint});

/// The official `gemma-4-E2B-it_qualcomm_qcs8275.litertlm`
/// (litert-community/gemma-4-E2B-it-litert-lm @b3ca0d2f, header v1.5.0,
/// copy in `test_assets/litertlm/gemma4_e2b_qualcomm_qcs8275.header.bin`):
/// the towers declared by the file, both on the CPU (the audio encoder and
/// adapter and the vision adapter `backend_constraint: cpu`, the vision
/// encoder unconstrained), prefill on the NPU.
const List<HeaderSection> kQualcommQcs8275Sections = [
  (type: 'tf_lite_embedder', constraint: null),
  (type: 'tf_lite_per_layer_embedder', constraint: null),
  (type: 'tf_lite_audio_encoder_hw', constraint: 'cpu'),
  (type: 'tf_lite_audio_adapter', constraint: 'cpu'),
  (type: 'tf_lite_end_of_audio', constraint: null),
  (type: 'tf_lite_vision_encoder', constraint: null),
  (type: 'tf_lite_vision_adapter', constraint: 'cpu'),
  (type: 'tf_lite_end_of_vision', constraint: null),
  (type: 'tf_lite_prefill_decode', constraint: 'npu'),
  (type: 'tf_lite_aux', constraint: null),
];

/// The official `gemma-4-E2B-it_qualcomm_sm8750.litertlm`
/// (litert-community/gemma-4-E2B-it-litert-lm @b3ca0d2f, header v1.5.0,
/// copy in `test_assets/litertlm/gemma4_e2b_qualcomm_sm8750.header.bin`):
/// both towers, no `backend_constraint` on any section, so where the towers
/// run is the user's choice on the NPU. Its towers are not compiled for the
/// NPU (on the NPU its load fails with "Failed to create conversation" at
/// the warm-up): the user sets them to the CPU. Google's all-NPU Gemma 4 E2B
/// builds (not public) share this undeclared layout, with towers that do run
/// on the NPU.
const List<HeaderSection> kQualcommSm8750Sections = [
  (type: 'tf_lite_embedder', constraint: null),
  (type: 'tf_lite_aux', constraint: null),
  (type: 'tf_lite_per_layer_embedder', constraint: null),
  (type: 'tf_lite_prefill_decode', constraint: null),
  (type: 'tf_lite_vision_encoder', constraint: null),
  (type: 'tf_lite_vision_adapter', constraint: null),
  (type: 'tf_lite_end_of_vision', constraint: null),
  (type: 'tf_lite_audio_encoder_hw', constraint: null),
  (type: 'tf_lite_audio_adapter', constraint: null),
  (type: 'tf_lite_end_of_audio', constraint: null),
];

/// The standard Gemma 4 E2B (`gemma-4-E2B-it.litertlm` with its MTP
/// drafter; header copy in `test_assets/litertlm/gemma4_e2b_mtp.header.bin`):
/// the audio encoder and adapter and the vision adapter pinned to the CPU,
/// nothing else declared.
const List<HeaderSection> kGemma4E2bSections = [
  (type: 'tf_lite_embedder', constraint: null),
  (type: 'tf_lite_per_layer_embedder', constraint: null),
  (type: 'tf_lite_audio_encoder_hw', constraint: 'cpu'),
  (type: 'tf_lite_audio_adapter', constraint: 'cpu'),
  (type: 'tf_lite_end_of_audio', constraint: null),
  (type: 'tf_lite_vision_encoder', constraint: null),
  (type: 'tf_lite_vision_adapter', constraint: 'cpu'),
  (type: 'tf_lite_end_of_vision', constraint: null),
  (type: 'tf_lite_prefill_decode', constraint: null),
  (type: 'tf_lite_mtp_drafter', constraint: null),
];

/// The older text-only per-SoC build (`gemma4_2b_SM8850.litertlm`, header
/// copy in `test_assets/litertlm/gemma4_2b_sm8850_npu.header.bin`): prefill
/// on the NPU, no towers.
const List<HeaderSection> kTextOnlyNpuSections = [
  (type: 'tf_lite_aux', constraint: null),
  (type: 'tf_lite_embedder', constraint: null),
  (type: 'tf_lite_per_layer_embedder', constraint: null),
  (type: 'tf_lite_prefill_decode', constraint: 'npu'),
];

/// Qwen3 0.6B (header copy in `test_assets/litertlm/qwen3_0_6b.header.bin`):
/// one section, no towers.
const List<HeaderSection> kQwen3Sections = [
  (type: 'tf_lite_prefill_decode', constraint: null),
];

/// [sections] as the app's header parser hands them on.
List<LitertlmSection> litertlmSections(List<HeaderSection> sections) => [
  for (final s in sections)
    LitertlmSection(modelType: s.type, backendConstraint: s.constraint),
];

/// The first bytes of a `.litertlm` whose header holds [sections], as
/// `parseLitertlmHeader` reads them: the 32-byte prefix (the magic, version
/// 1.5.0, the header end), then a `LiteRTLMMetaData` flatbuffer — the root
/// table's `section_metadata` (field 1), its `objects` (field 0), each
/// object's `items` (field 0), each item a key (field 0) and a
/// `StringValue` (field 2) whose string is field 0. Offsets only point
/// forward: every child is written after its parent.
Uint8List litertlmHeaderBytes(List<HeaderSection> sections) {
  final fb = _Flatbuffer();
  int item(String key, String value) => fb.table([
    () => fb.string(key),
    null,
    () => fb.table([() => fb.string(value)]),
  ]);
  int section(HeaderSection s) => fb.table([
    () => fb.vector([
      if (s.type case final type?) () => item('model_type', type),
      if (s.constraint case final constraint?)
        () => item('backend_constraint', constraint),
    ]),
  ]);
  fb.root(
    () => fb.table([
      null,
      () => fb.table([
        () => fb.vector([for (final s in sections) () => section(s)]),
      ]),
    ]),
  );
  final header = fb.finish();
  final prefix = ByteData(32);
  for (final (i, unit) in 'LITERTLM'.codeUnits.indexed) {
    prefix.setUint8(i, unit);
  }
  prefix
    ..setUint32(8, 1, Endian.little)
    ..setUint32(12, 5, Endian.little)
    ..setUint64(24, 32 + header.length, Endian.little);
  return Uint8List.fromList([...prefix.buffer.asUint8List(), ...header]);
}

/// A minimal flatbuffer writer: tables of offset fields, vectors of
/// offsets and strings, laid out breadth-first after a root offset.
final class _Flatbuffer {
  final List<int> _bytes = [];

  /// The offset slots still to fill: each child is written later, at the
  /// end, and its slot gets the distance to it.
  final List<(int, int Function())> _pending = [];

  int get _pos => _bytes.length;

  void _u16(int v) => _bytes
    ..add(v & 0xff)
    ..add((v >> 8) & 0xff);

  void _u32(int v) {
    for (var i = 0; i < 4; i++) {
      _bytes.add((v >> (8 * i)) & 0xff);
    }
  }

  void _align() {
    while (_bytes.length % 4 != 0) {
      _bytes.add(0);
    }
  }

  int _slot(int Function()? child) {
    final at = _pos;
    _u32(0);
    if (child != null) _pending.add((at, child));
    return at;
  }

  void root(int Function() table) => _slot(table);

  /// A table whose fields are offsets (null: absent); returns where its
  /// slots point (the table, after its vtable).
  int table(List<int Function()?> fields) {
    _align();
    final vtable = _pos;
    _u16(4 + 2 * fields.length);
    _u16(4 + 4 * fields.length);
    for (final (i, field) in fields.indexed) {
      _u16(field == null ? 0 : 4 + 4 * i);
    }
    _align();
    final table = _pos;
    _u32(table - vtable);
    for (final field in fields) {
      _slot(field);
    }
    return table;
  }

  int vector(List<int Function()> entries) {
    _align();
    final at = _pos;
    _u32(entries.length);
    for (final entry in entries) {
      _slot(entry);
    }
    return at;
  }

  int string(String value) {
    _align();
    final at = _pos;
    final units = utf8.encode(value);
    _u32(units.length);
    _bytes.addAll(units);
    return at;
  }

  Uint8List finish() {
    while (_pending.isNotEmpty) {
      final (slot, write) = _pending.removeAt(0);
      final target = write();
      final distance = target - slot;
      for (var i = 0; i < 4; i++) {
        _bytes[slot + i] = (distance >> (8 * i)) & 0xff;
      }
    }
    _align();
    return Uint8List.fromList(_bytes);
  }
}
