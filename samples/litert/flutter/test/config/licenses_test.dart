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

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/licenses.dart';

void main() {
  test('the built-in models are on the licence page: YOLO26n under AGPL-3.0 '
      'with its derivation script, EmbeddingGemma 2 under Apache-2.0 with its '
      'pinned repo', () {
    final yolo = modelLicenses.firstWhere(
      (e) => e.packages.single.startsWith('YOLO26n'),
    );
    final yoloText = yolo.paragraphs.map((p) => p.text).join('\n');
    expect(yoloText, contains('AGPL-3.0'));
    expect(yoloText, contains('tool/prune_yolo26n_head.py'));
    expect(yoloText, contains('Arm/yolo26n-fp16-litert'));

    final gemma = modelLicenses.firstWhere(
      (e) => e.packages.single.startsWith('EmbeddingGemma'),
    );
    final gemmaText = gemma.paragraphs.map((p) => p.text).join('\n');
    expect(gemmaText, contains('embeddinggemma-2-740m.litertlm'));
    expect(
      gemmaText,
      contains(
        'litert-community/embeddinggemma-2-740m-litert-lm at revision '
        '24d962e906c7d332c6428e71c9676855024569e2',
      ),
    );
    expect(gemmaText, contains('Apache License, Version 2.0'));
    expect(
      modelLicenses.map((e) => e.paragraphs.map((p) => p.text).join('\n')),
      everyElement(isNot(contains('Gemma Terms of Use'))),
      reason: 'no built-in model is under the Gemma terms any more',
    );
  });

  test('the app itself: Apache-2.0, first on the page; no AGPL for the app '
      'code', () {
    final app = modelLicenses.first;
    expect(app.packages.single, 'LiteRT Demos app');
    final text = app.paragraphs.map((p) => p.text).join('\n');
    expect(text, contains('LiteRT Demos app: Apache-2.0'));
    expect(text, contains('Copyright 2026 The Google AI Edge Authors'));
    expect(text, contains('http://www.apache.org/licenses/LICENSE-2.0'));
    expect(text, isNot(contains('AGPL')));
  });

  test('Inflect-nano-v2 names its litert-community repo, the one '
      'tool/models.lock fetches it from', () {
    final inflect = modelLicenses.firstWhere(
      (e) => e.packages.single.startsWith('Inflect-nano-v2'),
    );
    final text = inflect.paragraphs.map((p) => p.text).join('\n');
    expect(text, contains('litert-community/Inflect-Nano-v2'));
    expect(text, contains('Apache License, Version 2.0'));
  });

  test("the self-test's photo (CC BY-SA 2.0) is credited: title, author, "
      'source and licence', () {
    final cats = modelLicenses.firstWhere(
      (e) => e.packages.single.contains('cats.jpg'),
    );
    final text = cats.paragraphs.map((p) => p.text).join('\n');
    expect(text, contains('"Cats and remote controllers." by DocChewbacca'));
    expect(text, contains('https://www.flickr.com/photo.gne?id=210383891'));
    expect(text, contains('COCO 2017'));
    expect(text, contains('https://creativecommons.org/licenses/by-sa/2.0/'));
  });

  test('the repository LICENSE is the full Apache-2.0 text', () {
    final license = File('LICENSE').readAsStringSync();
    expect(license, contains('Apache License'));
    expect(license, contains('Version 2.0, January 2004'));
    expect(license, contains('END OF TERMS AND CONDITIONS'));
  });

  test('registered once in the LicenseRegistry', () async {
    registerModelLicenses();
    final entries = await LicenseRegistry.licenses.toList();
    expect(
      entries.expand((e) => e.packages),
      containsAll([
        'LiteRT Demos app',
        'YOLO26n detector (built in)',
        'EmbeddingGemma 2 740m (built in)',
      ]),
    );
  });
}
