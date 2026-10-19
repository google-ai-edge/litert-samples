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

/// What a `.litertlm` says about itself, for tests, without any model bytes:
/// [LiteRtLmBundleInfo]s built by hand with the plugin's public constructors,
/// and a reader ([fakeReadBundle]) that describes a test file by the tag it
/// starts with ([fakeLitertlmBytes]). No real header is committed.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show
        LiteRtLmBundleException,
        LiteRtLmBundleInfo,
        LiteRtLmBundleProblem,
        LiteRtLmNpuTarget,
        LiteRtLmSection;

import 'test_bytes.dart';

/// The NPU a per-SoC test bundle is compiled for.
const kTestNpuTarget = LiteRtLmNpuTarget(vendor: 'Qualcomm', soc: 'SM8750');

/// A `.litertlm` described as `readLiteRtLmBundleInfo` would describe it: a
/// text model, the encoders asked for, and, with [npu], a `tf_lite_aux`
/// section and a text model compiled for that NPU
/// ([LiteRtLmBundleInfo.isCompiledForNpu]); with [encodersOnNpu] too the
/// encoders carry that stamp (an all-NPU build). [visionConstraint] and
/// [audioConstraint] are an encoder's declared `backend_constraint`.
/// [fileLength] is the file's length; the sections end at [expectedLength]
/// (the file's length by default: complete).
LiteRtLmBundleInfo bundleInfo({
  bool vision = false,
  bool audio = false,
  LiteRtLmNpuTarget? npu,
  bool encodersOnNpu = false,
  String? visionConstraint,
  String? audioConstraint,
  int fileLength = 1 << 20,
  int? expectedLength,
}) {
  final end = expectedLength ?? fileLength;
  final types = [
    if (npu != null) 'tf_lite_aux',
    'tf_lite_prefill_decode',
    if (vision) 'tf_lite_vision_encoder',
    if (audio) 'tf_lite_audio_encoder_hw',
  ];
  final step = end ~/ (types.length + 1);
  return LiteRtLmBundleInfo(
    majorVersion: 1,
    minorVersion: 5,
    patchVersion: 0,
    systemMetadata: const {},
    fileLength: fileLength,
    sections: [
      for (final (i, type) in types.indexed)
        LiteRtLmSection(
          dataType: 'TFLiteModel',
          begin: step * (i + 1),
          end: i == types.length - 1 ? end : step * (i + 2),
          items: {
            'model_type': type,
            if (type == 'tf_lite_vision_encoder' && visionConstraint != null)
              'backend_constraint': visionConstraint,
            if (type == 'tf_lite_audio_encoder_hw' && audioConstraint != null)
              'backend_constraint': audioConstraint,
          },
          npuTarget:
              type == 'tf_lite_prefill_decode' ||
                  type == 'tf_lite_aux' ||
                  encodersOnNpu
              ? npu
              : null,
        ),
    ],
  );
}

/// The kinds of `.litertlm` the tests use. Where flutter_edge_ai_litertlm
/// 1.14.0 places each kind's towers when the app leaves them unset (README
/// "Where a model runs"): a declared `backend_constraint` (the CPU when it
/// allows it), the NPU for an encoder compiled for one, else the CPU.
enum FakeBundle {
  /// Gemma 4 E2B for the GPU and the CPU: both towers, no NPU; its towers
  /// run on the CPU.
  gemma4E2b(vision: true, audio: true),

  /// A text-only per-SoC NPU build: no towers, compiled for
  /// [kTestNpuTarget].
  textOnlyNpu(npu: kTestNpuTarget),

  /// An all-NPU build with both towers: the text model, `tf_lite_aux` and
  /// both encoders compiled for [kTestNpuTarget], so the plugin runs the
  /// towers on the NPU.
  npuWithTowers(
    vision: true,
    audio: true,
    npu: kTestNpuTarget,
    encodersOnNpu: true,
  ),

  /// The official `…_qualcomm_sm8750` shape: the text model and
  /// `tf_lite_aux` compiled for [kTestNpuTarget], the encoders not (no
  /// stamp), so the plugin runs the towers on the CPU.
  officialNpu(vision: true, audio: true, npu: kTestNpuTarget),

  /// Encoders that declare their backends (the Gemma 3n shape: vision
  /// `gpu`, audio `cpu`), so the plugin runs vision on the GPU and audio on
  /// the CPU.
  constrainedTowers(
    vision: true,
    audio: true,
    visionConstraint: 'gpu',
    audioConstraint: 'cpu',
  ),

  /// A text-only build for a Google Tensor NPU, which this package has no
  /// dispatch for: the plugin refuses every load of it (unsupportedNpu).
  tensorNpu(
    npu: LiteRtLmNpuTarget(vendor: 'Google', soc: 'Tensor_G5'),
  ),

  /// A text-only model for the GPU and the CPU (Qwen3 0.6B).
  textOnly(),

  /// Gemma 4 E2B cut short: its sections end past the end of the file.
  incomplete(vision: true, audio: true, complete: false);

  const FakeBundle({
    this.vision = false,
    this.audio = false,
    this.npu,
    this.encodersOnNpu = false,
    this.visionConstraint,
    this.audioConstraint,
    this.complete = true,
  });

  final bool vision;
  final bool audio;
  final LiteRtLmNpuTarget? npu;
  final bool encodersOnNpu;
  final String? visionConstraint;
  final String? audioConstraint;
  final bool complete;

  /// This kind for a file of [fileLength] bytes.
  LiteRtLmBundleInfo info(int fileLength) => bundleInfo(
    vision: vision,
    audio: audio,
    npu: npu,
    encodersOnNpu: encodersOnNpu,
    visionConstraint: visionConstraint,
    audioConstraint: audioConstraint,
    fileLength: fileLength,
    expectedLength: complete ? fileLength : fileLength * 2,
  );
}

const _tag = 'FAKE-LITERTLM:';

/// The bytes of a test `.litertlm` of [kind]: the tag [fakeReadBundle]
/// reads, then [filler] pseudo-random bytes.
Uint8List fakeLitertlmBytes(
  FakeBundle kind, {
  int filler = 8192,
  int seed = 3,
}) => Uint8List.fromList([
  ...ascii.encode('$_tag${kind.name}\n'),
  ...testBytes(filler, seed: seed),
]);

/// A `BundleReader` for tests, in place of `readLiteRtLmBundleInfo`: it
/// opens [path] as the plugin does (a missing or unreadable file is its
/// [FileSystemException]), then describes it by its tag
/// ([fakeLitertlmBytes]) with its length; a file without a tag is not a
/// `.litertlm`: the plugin's [LiteRtLmBundleProblem.notLitertlm], thrown as
/// the plugin's reader throws it (a load reads no header then).
Future<LiteRtLmBundleInfo> fakeReadBundle(String path) async {
  final file = await File(path).open();
  try {
    final length = await file.length();
    final head = ascii.decode(await file.read(64), allowInvalid: true);
    final kind = head.startsWith(_tag)
        ? FakeBundle.values
              .where(
                (k) => head.substring(_tag.length).startsWith('${k.name}\n'),
              )
              .firstOrNull
        : null;
    if (kind == null) {
      throw LiteRtLmBundleException(
        LiteRtLmBundleProblem.notLitertlm,
        '$path is not a .litertlm container (no LITERTLM magic)',
        path: path,
      );
    }
    return kind.info(length);
  } finally {
    await file.close();
  }
}

/// Writes a test `.litertlm` of [kind] at [path].
File writeFakeLitertlm(String path, FakeBundle kind, {int filler = 8192}) =>
    File(path)..writeAsBytesSync(fakeLitertlmBytes(kind, filler: filler));
