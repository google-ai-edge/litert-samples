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
    rejects(
      good.replaceFirst('"name":"g3_ekv1280.litertlm"', '"name":"../x"'),
      'file.name',
    );
  });

  group('tower settings of earlier builds: read and ignored', () {
    test('saved as version 4 with nothing about the towers', () {
      expect(CustomChatModelCodec.version, 4);
      final text = CustomChatModelCodec.encode(model);
      expect(text, contains('"v":4'));
      for (final key in [
        'supportImage',
        'supportAudio',
        'encoderBackend',
        'visionChoice',
        'audioChoice',
      ]) {
        expect(text, isNot(contains(key)));
      }
    });

    /// [current] (a version 4 record) as version [v] wrote it, with
    /// [legacy] (`"key":value` pairs) before `tools`.
    String asVersion(String current, int v, String legacy) => current
        .replaceFirst('"v":4', '"v":$v')
        .replaceFirst('"tools":', '$legacy,"tools":');

    test("a version 4 record's CPU | NPU choices decode, whatever their "
        'value, and are never written again', () {
      final v4 = CustomChatModelCodec.encode(model);
      for (final legacy in const [
        '"visionChoice":"cpu","audioChoice":"npu"',
        '"visionChoice":"gpu"',
        '"audioChoice":1',
        '"visionChoice":null',
      ]) {
        final back = CustomChatModelCodec.decode(asVersion(v4, 4, legacy));
        expect(back.backend, PreferredBackend.npu, reason: legacy);
        expect(back.maxTokens, 1280);
        expect(back.tools, isTrue);
        expect(CustomChatModelCodec.encode(back), v4, reason: legacy);
      }
    });

    test('version 1 to 3 records decode with their image, audio and encoder '
        'settings ignored, whatever their value', () {
      final gpu = model.copyWith(
        backend: PreferredBackend.gpu,
        maxTokens: 4096,
      );
      final v4 = CustomChatModelCodec.encode(gpu);
      for (final (v, legacy) in const [
        (1, '"supportImage":false'),
        (1, '"supportImage":"yes"'),
        (2, '"supportImage":true,"supportAudio":false'),
        // A version 3 build refused encoderBackend npu off the NPU.
        (3, '"supportImage":true,"supportAudio":1,"encoderBackend":"npu"'),
        (3, '"encoderBackend":"gpu"'),
      ]) {
        final back = CustomChatModelCodec.decode(asVersion(v4, v, legacy));
        expect(back.backend, PreferredBackend.gpu, reason: legacy);
        expect(back.maxTokens, 4096);
        expect(CustomChatModelCodec.encode(back), v4, reason: legacy);
      }
    });

    test('the record a version 3 build wrote for an NPU build with images '
        'off and the encoders on the NPU decodes: nothing of the towers '
        'remains', () {
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
      final saved = CustomChatModelCodec.encode(back);
      expect(saved, contains('"v":4'));
      expect(saved, isNot(contains('supportImage')));
      expect(saved, isNot(contains('encoderBackend')));
    });
  });
}
