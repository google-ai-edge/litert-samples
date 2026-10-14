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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/model_towers.dart';

import '../../support/litertlm_header.dart';

const _cpu = PreferredBackend.cpu;
const _gpu = PreferredBackend.gpu;
const _npu = PreferredBackend.npu;

/// `vision cpu (declared) · audio npu (choice)`: each tower with its
/// backend and reason, `-` when the file has none.
String _placed(ModelTowers towers) => [
  'vision ${towers.vision ?? '-'}',
  'audio ${towers.audio ?? '-'}',
].join(' · ');

ModelTowers _on(
  List<HeaderSection> header,
  PreferredBackend backend, {
  PreferredBackend? vision,
  PreferredBackend? audio,
}) => resolveTowers(
  litertlmSections(header),
  backend,
  vision: vision,
  audio: audio,
);

void main() {
  group('the four real headers of the task', () {
    test('towers declared by the file (the official qcs8275 build): on the '
        'NPU, Text NPU, Vision CPU, Audio CPU, both set by the file', () {
      final towers = _on(kQualcommQcs8275Sections, _npu);
      expect(_placed(towers), 'vision cpu (declared) · audio cpu (declared)');
      expect(towers.parts, ['vision cpu', 'audio cpu']);
      expect(towers.undeclaredOnNpu, isFalse);
      expect(
        _placed(_on(kQualcommQcs8275Sections, _npu, vision: _npu, audio: _npu)),
        'vision cpu (declared) · audio cpu (declared)',
        reason: 'a declaration always wins over a saved choice',
      );
    });

    test('undeclared towers (the official sm8750 build): on the NPU both '
        'towers are a choice, the NPU by default; on the GPU both run on the '
        'CPU', () {
      final npu = _on(kQualcommSm8750Sections, _npu);
      expect(_placed(npu), 'vision npu (choice) · audio npu (choice)');
      expect(npu.undeclaredOnNpu, isTrue);
      expect(
        _placed(_on(kQualcommSm8750Sections, _gpu)),
        'vision cpu (offNpu) · audio cpu (offNpu)',
      );
      expect(
        _placed(_on(kQualcommSm8750Sections, _cpu)),
        'vision cpu (offNpu) · audio cpu (offNpu)',
      );
    });

    test(
      'standard Gemma 4 E2B: on the GPU, Vision CPU and Audio CPU (both '
      'set by the file: the vision adapter and the audio sections say cpu)',
      () {
        final towers = _on(kGemma4E2bSections, _gpu);
        expect(_placed(towers), 'vision cpu (declared) · audio cpu (declared)');
        expect(towers.supportImage && towers.supportAudio, isTrue);
        expect(
          _placed(_on(kGemma4E2bSections, _npu)),
          'vision cpu (declared) · audio cpu (declared)',
        );
      },
    );

    test('text-only files (the older SM8850 build, Qwen3): no vision, no '
        'audio, on any backend', () {
      for (final header in [kTextOnlyNpuSections, kQwen3Sections]) {
        for (final backend in PreferredBackend.values) {
          final towers = _on(header, backend);
          expect(towers, ModelTowers.none, reason: '$header on $backend');
          expect(towers.supportImage || towers.supportAudio, isFalse);
          expect(towers.parts, isEmpty);
        }
      }
    });
  });

  group('the official Qualcomm builds the coordinator checked', () {
    test('sm8750 declares nothing: on the NPU two choices defaulting to the '
        'NPU (its towers are not compiled for it: the user sets CPU); set to '
        'the CPU they load there', () {
      expect(
        _placed(_on(kQualcommSm8750Sections, _npu)),
        'vision npu (choice) · audio npu (choice)',
      );
      final cpu = _on(kQualcommSm8750Sections, _npu, vision: _cpu, audio: _cpu);
      expect(_placed(cpu), 'vision cpu (choice) · audio cpu (choice)');
      expect(cpu.undeclaredOnNpu, isFalse);
      expect(cpu.visionBackend, _cpu);
      expect(cpu.audioBackend, _cpu);
    });

    test('qcs8275 pins its towers to the CPU: no choice on the NPU', () {
      expect(
        _placed(_on(kQualcommQcs8275Sections, _npu)),
        'vision cpu (declared) · audio cpu (declared)',
      );
    });
  });

  group('the rule, tower by tower', () {
    test('one tower declared, the other not: only the open one is a choice, '
        'and a choice moves only its own tower', () {
      const mixed = [
        (type: 'tf_lite_vision_encoder', constraint: null),
        (type: 'tf_lite_vision_adapter', constraint: 'cpu'),
        (type: 'tf_lite_audio_encoder_hw', constraint: null),
        (type: 'tf_lite_audio_adapter', constraint: null),
      ];
      expect(
        _placed(_on(mixed, _npu)),
        'vision cpu (declared) · audio npu (choice)',
      );
      expect(
        _placed(_on(mixed, _npu, audio: _cpu)),
        'vision cpu (declared) · audio cpu (choice)',
      );
      expect(
        _placed(_on(mixed, _npu, vision: _npu)),
        'vision cpu (declared) · audio npu (choice)',
      );
    });

    test('cpu wins over npu inside a tower; a tower declared npu runs on '
        'the NPU on any backend', () {
      expect(
        _placed(
          _on(const [
            (type: 'tf_lite_vision_encoder', constraint: 'npu'),
            (type: 'tf_lite_vision_adapter', constraint: 'cpu'),
          ], _npu),
        ),
        'vision cpu (declared) · audio -',
      );
      const npuPinned = [
        (type: 'tf_lite_audio_encoder_hw', constraint: 'npu'),
        (type: 'tf_lite_audio_adapter', constraint: null),
      ];
      for (final backend in PreferredBackend.values) {
        expect(
          _placed(_on(npuPinned, backend, audio: _cpu)),
          'vision - · audio npu (declared)',
          reason: '$backend',
        );
      }
    });

    test('a tower declared gpu runs on the GPU, read-only (never the CPU it '
        'cannot run on); a value that names no backend is no declaration', () {
      const gpuPinned = [
        (type: 'tf_lite_vision_encoder', constraint: 'gpu'),
        (type: 'tf_lite_vision_adapter', constraint: null),
      ];
      for (final backend in PreferredBackend.values) {
        expect(
          _placed(_on(gpuPinned, backend, vision: _cpu)),
          'vision gpu (declared) · audio -',
          reason: '$backend',
        );
      }
      const odd = [(type: 'tf_lite_audio_encoder_hw', constraint: 'tpu')];
      expect(_placed(_on(odd, _npu)), 'vision - · audio npu (choice)');
      expect(_placed(_on(odd, _gpu)), 'vision - · audio cpu (offNpu)');
    });

    test('the end-of markers and the decoder do not count, an adapter alone '
        'is no tower', () {
      expect(
        _placed(
          _on(const [
            (type: 'tf_lite_vision_encoder', constraint: null),
            (type: 'tf_lite_end_of_vision', constraint: 'cpu'),
            (type: 'tf_lite_audio_encoder_hw', constraint: null),
            (type: 'tf_lite_end_of_audio', constraint: 'cpu'),
            (type: 'tf_lite_prefill_decode', constraint: 'cpu'),
          ], _npu),
        ),
        'vision npu (choice) · audio npu (choice)',
      );
      expect(
        _on(const [
          (type: 'tf_lite_vision_adapter', constraint: 'cpu'),
          (type: 'tf_lite_audio_adapter', constraint: 'cpu'),
        ], _npu),
        ModelTowers.none,
      );
    });

    test(
      'off the NPU a choice means nothing: an open tower runs on the CPU',
      () {
        expect(
          _placed(
            _on(kQualcommSm8750Sections, _gpu, vision: _npu, audio: _npu),
          ),
          'vision cpu (offNpu) · audio cpu (offNpu)',
        );
      },
    );

    test('towerOf and hasTower', () {
      expect(towerOf('tf_lite_vision_encoder'), Tower.vision);
      expect(towerOf('tf_lite_vision_adapter'), Tower.vision);
      expect(towerOf('tf_lite_audio_encoder_hw'), Tower.audio);
      expect(towerOf('tf_lite_audio_adapter'), Tower.audio);
      expect(towerOf('tf_lite_end_of_audio'), isNull);
      expect(towerOf('tf_lite_end_of_vision'), isNull);
      expect(towerOf('tf_lite_prefill_decode'), isNull);
      expect(towerOf(null), isNull);
      final all = litertlmSections(kQualcommSm8750Sections);
      expect(hasTower(all, Tower.vision) && hasTower(all, Tower.audio), isTrue);
      expect(hasTower(litertlmSections(kQwen3Sections), Tower.vision), isFalse);
    });
  });
}
