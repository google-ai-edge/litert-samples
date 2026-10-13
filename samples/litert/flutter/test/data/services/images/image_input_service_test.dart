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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker_platform_interface/image_picker_platform_interface.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/image_repository.dart';
import 'package:litert_edge_demos/data/services/images/image_input_service.dart';
import 'package:litert_edge_demos/data/services/images/image_normalizer.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// The plugin's platform side, like `image_picker_macos` without a camera
/// delegate unless [camera] is set.
class _FakePickerPlatform extends ImagePickerPlatform {
  _FakePickerPlatform({this.camera = false});

  final bool camera;
  final List<(ImageSource, ImagePickerOptions)> calls = [];
  XFile? next;

  @override
  bool supportsImageSource(ImageSource source) =>
      source == ImageSource.gallery || camera;

  @override
  Future<XFile?> getImageFromSource({
    required ImageSource source,
    ImagePickerOptions options = const ImagePickerOptions(),
  }) async {
    calls.add((source, options));
    return next;
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late ImagePickerPlatform original;
  setUp(() => original = ImagePickerPlatform.instance);
  tearDown(() => ImagePickerPlatform.instance = original);

  test('gallery: asks the plugin for ≤1024 px without full metadata and '
      'returns the bytes', () async {
    final platform = _FakePickerPlatform()
      ..next = XFile.fromData(Uint8List.fromList([1, 2, 3]));
    ImagePickerPlatform.instance = platform;
    final service = PlatformImageInputService();

    final bytes = await service.pick(ImageSourceKind.gallery);

    expect(bytes, [1, 2, 3]);
    final (source, options) = platform.calls.single;
    expect(source, ImageSource.gallery);
    expect(options.maxWidth, kLlmImageMaxSide);
    expect(options.maxHeight, kLlmImageMaxSide);
    expect(options.requestFullMetadata, isFalse);
  });

  test('a cancelled picker returns null', () async {
    ImagePickerPlatform.instance = _FakePickerPlatform();

    expect(
      await PlatformImageInputService().pick(ImageSourceKind.gallery),
      isNull,
    );
  });

  test('camera without platform support (macOS) is refused before the '
      'plugin, with a message that says what to do', () async {
    final platform = _FakePickerPlatform();
    ImagePickerPlatform.instance = platform;
    final service = PlatformImageInputService();

    expect(service.supports(ImageSourceKind.camera), isFalse);
    await expectLater(
      service.pick(ImageSourceKind.camera),
      throwsA(
        isA<UnsupportedImageSourceException>().having(
          (e) => e.toString(),
          'message',
          contains('gallery'),
        ),
      ),
    );
    expect(platform.calls, isEmpty);
  });

  test('camera where the platform has one (iOS, Android)', () async {
    final platform = _FakePickerPlatform(camera: true)
      ..next = XFile.fromData(Uint8List.fromList([9]));
    ImagePickerPlatform.instance = platform;

    final bytes = await PlatformImageInputService().pick(
      ImageSourceKind.camera,
    );

    expect(bytes, [9]);
    expect(platform.calls.single.$1, ImageSource.camera);
  });

  test('ImageRepository over the real service and normalizer: a real JPEG '
      'from the plugin becomes the PNG the chat gets', () async {
    final jpeg = await XFile('test_assets/cats.jpg').readAsBytes();
    ImagePickerPlatform.instance = _FakePickerPlatform()
      ..next = XFile.fromData(jpeg);
    final repository = ImageRepository(input: PlatformImageInputService());

    final result = await repository.pick(ImageSourceKind.gallery);

    final image = (result as Ok<LlmImage?>).value!;
    expect((image.width, image.height), (640, 480));
    expect(image.sourceBytes, jpeg.length);
    expect(image.png.sublist(0, 4), [0x89, 0x50, 0x4E, 0x47]);
    expect(image.normalizeTime, greaterThan(Duration.zero));
  });

  test('ImageRepository: an undecodable file is an error, not an empty '
      'attachment', () async {
    ImagePickerPlatform.instance = _FakePickerPlatform()
      ..next = XFile.fromData(Uint8List.fromList(List.filled(32, 1)));
    final repository = ImageRepository(input: PlatformImageInputService());

    final result = await repository.pick(ImageSourceKind.gallery);

    expect(
      (result as Error<LlmImage?>).error,
      isA<UndecodableImageException>(),
    );
  });
}
