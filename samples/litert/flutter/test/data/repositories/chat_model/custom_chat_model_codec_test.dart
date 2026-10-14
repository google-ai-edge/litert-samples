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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/chat_model/custom_chat_model_codec.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';

import '../../../support/litertlm_header.dart';

void main() {
  final sha = 'a1' * 32;
  final model = CustomChatModel(
    displayName: 'Gemma 3 1B NPU',
    source: UrlModelSource(
      Uri.parse('https://example.com/g3_ekv1280.litertlm'),
      sha256: sha,
      sizeBytes: 1234,
    ),
    file: CustomModelFile(
      name: 'g3_ekv1280.litertlm',
      sizeBytes: 1234,
      sha256: sha,
      checksumMatched: true,
    ),
    modelType: ModelType.gemmaIt,
    backend: PreferredBackend.npu,
    maxTokens: 1280,
    tools: true,
  );

  test('round-trips through its saved JSON', () {
    final back = CustomChatModelCodec.decode(
      CustomChatModelCodec.encode(model),
    );
    expect(back.displayName, model.displayName);
    expect(back.modelType, ModelType.gemmaIt);
    expect(back.backend, PreferredBackend.npu);
    expect(back.maxTokens, 1280);
    expect(back.tools, isTrue);
    expect(back.visionChoice, isNull);
    expect(back.audioChoice, isNull);
    final source = back.source as UrlModelSource;
    expect(source.url, Uri.parse('https://example.com/g3_ekv1280.litertlm'));
    expect(source.sha256, sha);
    expect(source.sizeBytes, 1234);
    expect(back.file?.sha256, sha);
    expect(back.file?.checksumMatched, isTrue);
  });

  test("a link's query and fragment (a signed link's token) are never "
      'saved', () {
    final signed = model.copyWith(
      source: UrlModelSource(
        Uri.parse(
          'https://cdn.example.com/g3.litertlm?X-Signature=secret#part',
        ),
      ),
    );
    final text = CustomChatModelCodec.encode(signed);
    expect(text, isNot(contains('secret')));
    expect(text, isNot(contains('#part')));
    expect(
      (CustomChatModelCodec.decode(text).source as UrlModelSource).url,
      Uri.parse('https://cdn.example.com/g3.litertlm'),
    );
  });

  test('a whole link an earlier build saved is read without its user, '
      'query and fragment', () {
    final saved = CustomChatModelCodec.encode(model).replaceFirst(
      'https://example.com/g3_ekv1280.litertlm',
      'https://me:hf_token@example.com/g3_ekv1280.litertlm?token=t#f',
    );
    expect(saved, contains('hf_token'), reason: 'the old form');
    final back = CustomChatModelCodec.decode(saved);
    final url = (back.source as UrlModelSource).url;
    expect(url, Uri.parse('https://example.com/g3_ekv1280.litertlm'));
    expect(CustomChatModelCodec.encode(back), isNot(contains('hf_token')));
    expect(back.sourceLine, isNot(contains('token')));
  });

  test('an imported model without a file yet round-trips too', () {
    const imported = CustomChatModel(
      displayName: 'mine',
      source: ImportedModelSource('/Users/me/Downloads/mine.litertlm'),
      backend: PreferredBackend.gpu,
      maxTokens: 4096,
    );
    final back = CustomChatModelCodec.decode(
      CustomChatModelCodec.encode(imported),
    );
    expect(
      (back.source as ImportedModelSource).pickedPath,
      '/Users/me/Downloads/mine.litertlm',
    );
    expect(back.file, isNull);
  });

  test('decoding is strict: every problem is named, nothing defaulted', () {
    void rejects(String json, String why) => expect(
      () => CustomChatModelCodec.decode(json),
      throwsA(
        isA<FormatException>().having(
          (e) => e.message,
          'message',
          contains(why),
        ),
      ),
    );
    final good = CustomChatModelCodec.encode(model);
    rejects('{', 'not JSON');
    rejects(good.replaceFirst('"v":4', '"v":5'), 'version 5');
    rejects(good.replaceFirst('"v":4', '"v":0'), 'version 0');
    rejects(good.replaceFirst('"v":4', '"v":"4"'), 'version 4');
    rejects(good.replaceFirst('"backend":"npu"', '"backend":"tpu"'), 'backend');
    rejects(
      good.replaceFirst('"modelType":"gemmaIt"', '"modelType":"gemma9"'),
      'modelType',
    );
    rejects(
      good.replaceFirst('"maxTokens":1280', '"maxTokens":64'),
      'maxTokens',
    );
    rejects(
      good.replaceFirst('"tools":true', '"tools":"yes"'),
      'tools must be a bool',
    );
    rejects(
      good.replaceFirst('"v":4', '"v":4,"extra":0'),
      'unknown key "extra"',
    );
    // Version 4 no longer has the settings the header decides.
    for (final key in ['supportImage', 'supportAudio', 'encoderBackend']) {
      rejects(
        good.replaceFirst('"v":4', '"v":4,"$key":false'),
        'unknown key "$key"',
      );
    }
    rejects(
      good.replaceFirst('"name":"g3_ekv1280.litertlm"', '"name":"../x"'),
      'file.name',
    );
  });

  group("the towers' choices (version 4)", () {
    test('saved as version 4 with only the choices that were made; cpu and '
        'npu round-trip, none is the NPU by default', () {
      expect(CustomChatModelCodec.version, 4);
      final text = CustomChatModelCodec.encode(model);
      expect(text, contains('"v":4'));
      expect(text, isNot(contains('Choice')), reason: 'no choice made');
      for (final key in ['supportImage', 'supportAudio', 'encoderBackend']) {
        expect(text, isNot(contains(key)));
      }

      final chosen = model.withTowerChoices(
        vision: PreferredBackend.cpu,
        audio: PreferredBackend.npu,
      );
      final chosenText = CustomChatModelCodec.encode(chosen);
      expect(chosenText, contains('"visionChoice":"cpu"'));
      expect(chosenText, contains('"audioChoice":"npu"'));
      final back = CustomChatModelCodec.decode(chosenText);
      expect(back.visionChoice, PreferredBackend.cpu);
      expect(back.audioChoice, PreferredBackend.npu);
    });

    test('a choice that is not cpu or npu is refused', () {
      final good = CustomChatModelCodec.encode(model);
      for (final (value, why) in const [
        ('"gpu"', 'visionChoice "gpu" is not one of cpu, npu'),
        ('"tpu"', 'visionChoice "tpu"'),
        ('1', 'visionChoice "1"'),
      ]) {
        expect(
          () => CustomChatModelCodec.decode(
            good.replaceFirst('"v":4', '"v":4,"visionChoice":$value'),
          ),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              contains(why),
            ),
          ),
        );
      }
    });

    test('an old saved version 3 record with supportImage false and '
        'encoderBackend npu decodes: the image and audio settings are '
        "ignored, the header's towers win, the encoders' NPU is no choice", () {
      // Exactly what a version 3 build wrote for an NPU build with images off
      // and the encoders on the NPU (Google's all-NPU builds were saved so).
      const v3 =
          '{"v":3,"displayName":"NPU build","modelType":"gemma4",'
          '"backend":"npu","maxTokens":4096,"supportImage":false,'
          '"supportAudio":true,"encoderBackend":"npu","tools":false,'
          '"source":{"kind":"local","path":"/data/local/tmp/litert-models/'
          'gemma4_e2b_npu.litertlm"},"file":{"name":'
          '"gemma4_e2b_npu.litertlm","sizeBytes":3000000000,'
          '"checksumMatched":false}}';

      final back = CustomChatModelCodec.decode(v3);

      expect(back.displayName, 'NPU build');
      expect(back.backend, PreferredBackend.npu);
      expect(back.maxTokens, 4096);
      expect(back.tools, isFalse);
      expect(back.visionChoice, isNull);
      expect(back.audioChoice, isNull);
      // The derived values win: an undeclared file's towers (the official
      // sm8750 build) both load, on the NPU by default.
      final undeclared = back.towersOf(
        litertlmSections(kQualcommSm8750Sections),
      );
      expect(
        undeclared.supportImage,
        isTrue,
        reason: 'supportImage false ignored',
      );
      expect(undeclared.visionBackend, PreferredBackend.npu);
      expect(undeclared.audioBackend, PreferredBackend.npu);
      // Towers declared cpu by the file (the official qcs8275 build): the
      // saved encoderBackend npu cannot put them on the NPU.
      final declared = back.towersOf(
        litertlmSections(kQualcommQcs8275Sections),
      );
      expect(declared.visionBackend, PreferredBackend.cpu);
      expect(declared.audioBackend, PreferredBackend.cpu);
      // A text-only file: no towers, whatever was saved.
      expect(
        back.towersOf(litertlmSections(kTextOnlyNpuSections)).supportAudio,
        isFalse,
        reason: 'supportAudio true ignored',
      );

      final saved = CustomChatModelCodec.encode(back);
      expect(saved, contains('"v":4'));
      expect(saved, isNot(contains('supportImage')));
      expect(saved, isNot(contains('encoderBackend')));
    });

    /// [current] (a version 4 record) as version [v] wrote it, with
    /// [legacy] (`"key":value` pairs) before `tools`.
    String asVersion(String current, int v, String legacy) => current
        .replaceFirst('"v":4', '"v":$v')
        .replaceFirst('"tools":', '$legacy,"tools":');

    test('version 1 and 2 records off the NPU still decode, their image '
        'and audio settings ignored, with no choices', () {
      final gpu = model.copyWith(
        backend: PreferredBackend.gpu,
        maxTokens: 4096,
      );
      final v4 = CustomChatModelCodec.encode(gpu);
      final v2 = asVersion(v4, 2, '"supportImage":true,"supportAudio":false');
      final v1 = asVersion(v4, 1, '"supportImage":false');
      for (final old in [v1, v2]) {
        final back = CustomChatModelCodec.decode(old);
        expect(back.backend, PreferredBackend.gpu);
        expect(back.maxTokens, 4096);
        expect(back.tools, isTrue);
        expect(back.visionChoice, isNull);
        expect(back.audioChoice, isNull);
        expect(CustomChatModelCodec.encode(back), v4);
      }
    });

    test('version 1 and 2 records on the NPU ran their towers on the CPU '
        '(no encoder setting existed): both choices decode as cpu, so the '
        'official qualcomm_sm8750 build (no tower declared) keeps its towers '
        'on the CPU', () {
      final v4 = CustomChatModelCodec.encode(model);
      final v2 = asVersion(v4, 2, '"supportImage":true,"supportAudio":true');
      final v1 = asVersion(v4, 1, '"supportImage":true');
      for (final old in [v1, v2]) {
        final back = CustomChatModelCodec.decode(old);
        expect(back.backend, PreferredBackend.npu);
        expect(back.visionChoice, PreferredBackend.cpu);
        expect(back.audioChoice, PreferredBackend.cpu);
        final towers = back.towersOf(litertlmSections(kQualcommSm8750Sections));
        expect(towers.visionBackend, PreferredBackend.cpu);
        expect(towers.audioBackend, PreferredBackend.cpu);
        expect(towers.undeclaredOnNpu, isFalse);
        final saved = CustomChatModelCodec.encode(back);
        expect(saved, contains('"v":4'));
        expect(saved, contains('"visionChoice":"cpu"'));
        expect(saved, contains('"audioChoice":"cpu"'));
      }
    });

    test('a version 3 record keeps where its encoders ran: encoderBackend cpu '
        'decodes as both choices cpu, npu as no choice (the NPU)', () {
      final v4 = CustomChatModelCodec.encode(model);
      const legacy = '"supportImage":true,"supportAudio":true';
      final cpu = CustomChatModelCodec.decode(
        asVersion(v4, 3, '$legacy,"encoderBackend":"cpu"'),
      );
      expect(cpu.visionChoice, PreferredBackend.cpu);
      expect(cpu.audioChoice, PreferredBackend.cpu);
      final towers = cpu.towersOf(litertlmSections(kQualcommSm8750Sections));
      expect(towers.visionBackend, PreferredBackend.cpu);
      expect(towers.audioBackend, PreferredBackend.cpu);

      final npu = CustomChatModelCodec.decode(
        asVersion(v4, 3, '$legacy,"encoderBackend":"npu"'),
      );
      expect(npu.visionChoice, isNull);
      expect(npu.audioChoice, isNull);
    });

    test('a damaged old record is the invalid-settings error, not accepted: '
        'each field its version required must be there with its type', () {
      final v4 = CustomChatModelCodec.encode(model);
      void rejects(String json, String why) => expect(
        () => CustomChatModelCodec.decode(json),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains(why),
          ),
        ),
      );
      const both = '"supportImage":true,"supportAudio":true';
      // Missing.
      rejects(v4.replaceFirst('"v":4', '"v":1'), 'supportImage');
      rejects(asVersion(v4, 2, '"supportImage":true'), 'supportAudio');
      rejects(asVersion(v4, 3, both), 'encoderBackend');
      // Mistyped.
      rejects(asVersion(v4, 1, '"supportImage":"yes"'), 'supportImage');
      rejects(
        asVersion(v4, 2, '"supportImage":true,"supportAudio":1'),
        'supportAudio',
      );
      rejects(
        asVersion(v4, 3, '$both,"encoderBackend":"gpu"'),
        'encoderBackend "gpu" is not one of cpu, npu',
      );
      rejects(
        asVersion(v4, 3, '$both,"encoderBackend":null'),
        'encoderBackend',
      );
      // Version 3 kept the encoders on the CPU off the NPU.
      final gpu = CustomChatModelCodec.encode(
        model.copyWith(backend: PreferredBackend.gpu, maxTokens: 4096),
      );
      rejects(
        asVersion(gpu, 3, '$both,"encoderBackend":"npu"'),
        'encoderBackend npu needs the NPU backend',
      );
    });

    test('an old record keeps its own keys: a choice in version 3, or '
        'encoderBackend in version 2, is unknown', () {
      final v4 = CustomChatModelCodec.encode(
        model.withTowerChoices(vision: PreferredBackend.cpu, audio: null),
      );
      expect(
        () => CustomChatModelCodec.decode(v4.replaceFirst('"v":4', '"v":3')),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('unknown key "visionChoice"'),
          ),
        ),
      );
      final v2 = CustomChatModelCodec.encode(model)
          .replaceFirst('"v":4', '"v":2,"encoderBackend":"cpu"');
      expect(
        () => CustomChatModelCodec.decode(v2),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('unknown key "encoderBackend"'),
          ),
        ),
      );
    });
  });
}
