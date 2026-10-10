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
import 'package:litert_edge_demos/utils/redact_url.dart';

void main() {
  test('drops user-info, query and fragment; keeps scheme, host, port and '
      'path', () {
    expect(
      redactUrl(
        Uri.parse(
          'https://user:token@huggingface.co:8443/org/repo/resolve/main/'
          'model.litertlm?download=true&token=hf_secret#frag',
        ),
      ),
      'https://huggingface.co:8443/org/repo/resolve/main/model.litertlm',
    );
  });

  test('a network camera URL with Basic credentials keeps host and '
      'port', () {
    final redacted = redactUrl(
      Uri.parse('http://admin:secret@192.168.1.23:8080/video?login=x'),
    );
    expect(redacted, 'http://192.168.1.23:8080/video');
    expect(redacted, isNot(contains('secret')));
    expect(redacted, isNot(contains('admin')));
  });

  test('a URL with nothing secret is unchanged', () {
    for (final url in [
      'https://example.com/model.litertlm',
      'http://127.0.0.1:8080/video',
      'https://[::1]:8443/p',
      'https://example.com',
    ]) {
      expect(redactUrl(Uri.parse(url)), url);
    }
  });

  test('percent-escapes in the path stay as they are', () {
    expect(
      redactUrl(Uri.parse('https://example.com/a%20b/c.litertlm?x=1')),
      'https://example.com/a%20b/c.litertlm',
    );
  });

  test('the default port stays implicit', () {
    expect(
      redactUrl(Uri.parse('https://example.com:443/m?sig=1')),
      'https://example.com/m',
    );
  });
}
