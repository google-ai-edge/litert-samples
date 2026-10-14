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

// The knowledge base's golden set: test_assets/kb_golden.json's questions.
// kb_retrieval_test.dart asks them typed, and spoken by the app's own TTS
// into memory at start-up (support/question_audio.dart): the spoken clips
// are not shipped.

/// The golden questions as test_assets/kb_golden.json lists them: on-topic
/// with their expected documents, off-topic questions, and everyday chat (what
/// a voice chat hears between knowledge questions: greetings, thanks, small
/// requests), which no gate may let through either.
typedef KbGolden = ({
  List<({String q, String doc})> onTopic,
  List<String> offTopic,
  List<String> everydayChat,
});

/// Parses test_assets/kb_golden.json. Throws a [FormatException] for an
/// entry of another shape.
KbGolden parseKbGolden(Object? json) {
  if (json case {
    'on_topic': final List<Object?> on,
    'off_topic': final List<Object?> off,
    'everyday_chat': final List<Object?> chat,
  }) {
    return (
      onTopic: [
        for (final entry in on)
          if (entry case {'q': final String q, 'expect_doc': final String doc})
            (q: q, doc: doc)
          else
            throw FormatException('Unexpected golden entry', entry),
      ],
      offTopic: _strings(off),
      everydayChat: _strings(chat),
    );
  }
  throw FormatException('Not the golden set', json);
}

List<String> _strings(List<Object?> list) => [
  for (final q in list)
    if (q is String) q else throw FormatException('Unexpected', q),
];
