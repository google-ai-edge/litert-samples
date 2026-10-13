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

/// The app's own licence and those of the models and the photo built into it,
/// shown on the licence page (Home › More › Licences) next to the packages'
/// own. The full texts are in
/// `LICENSE`, `assets/models/NOTICE.md`, `test_assets/README.md` and at the
/// links.
void registerModelLicenses() {
  LicenseRegistry.addLicense(() => Stream.fromIterable(modelLicenses));
}

/// The entries [registerModelLicenses] adds: the app first, then the models,
/// the self-test's photo and the Linux JPEG library.
const modelLicenses = [
  LicenseEntryWithLineBreaks(['LiteRT Demos app'], _appNotice),
  LicenseEntryWithLineBreaks(['YOLO26n detector (built in)'], _yolo26nNotice),
  LicenseEntryWithLineBreaks(['EmbeddingGemma 300M (built in)'], _gemmaNotice),
  LicenseEntryWithLineBreaks(['Whisper base (built in)'], _whisperNotice),
  LicenseEntryWithLineBreaks(['moonshine-tiny (built in)'], _moonshineNotice),
  LicenseEntryWithLineBreaks([
    'Inflect-nano-v2 TTS (built in)',
  ], _inflectNotice),
  LicenseEntryWithLineBreaks([
    'Self-test photo (test_assets/cats.jpg, built in)',
  ], _catsNotice),
  LicenseEntryWithLineBreaks([
    'libjpeg-turbo (Linux: lib/libturbojpeg.so.0)',
  ], _libjpegTurboNotice),
];

const _appNotice = '''
LiteRT Demos app: Apache-2.0.

Copyright 2026 The Google AI Edge Authors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this app's source code except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

The models built into the app and the packages it uses keep their own licences, listed on this page.''';

const _gemmaNotice = '''
The knowledge base's embedding model built into this app, EmbeddingGemma-300M (embeddinggemma-300M_seq512_mixed-precision.tflite, 179132472 bytes, SHA-256 ad09e81557203cb0e177abf9bf8727dfe138a7d394aa0f70f0b2ed16432e121a) with its tokenizer (sentencepiece.model, 4683319 bytes, SHA-256 d6daa52d93d7aad10e8388bd526c4e501d914b47177398d1d9621f1fe48438c7), from litert-community/embeddinggemma-300m, is redistributed unmodified.

Gemma is provided under and subject to the Gemma Terms of Use found at https://ai.google.dev/gemma/terms

Use of the model is also subject to the Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy''';

const _yolo26nNotice = '''
The object detector built into this app, assets/models/yolo26n_fp16_rawhead.tflite (10361332 bytes, SHA-256 5ddd5eebad18587d56500a30b0995c07c9e1a241640750f568e4a974f66ed80a), is a derivative of YOLO26n by Ultralytics, converted to LiteRT by Arm (Hugging Face: Arm/yolo26n-fp16-litert).

It was derived with tool/prune_yolo26n_head.py in this app's source repository: the detection head is cut at the [1, 8400, 84] tensor so the whole graph runs on the GPU. No weights were changed.

Ultralytics YOLO26 and the Arm conversion are licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This derivative is distributed under the same licence. You may obtain the corresponding source, including the derivation script, from the app's source repository. Full licence text: https://www.gnu.org/licenses/agpl-3.0.html

Ultralytics: https://github.com/ultralytics/ultralytics
Arm conversion: https://huggingface.co/Arm/yolo26n-fp16-litert''';

const _whisperNotice = '''
The speech recognizer of Voice chat built into this app, Whisper base int8 (whisper_base_30s_i8.tflite, 77012960 bytes, SHA-256 f6943d9d293138850b729e074057956c664891c57837692b1bac4608c4506cd1, Hugging Face: litert-community/whisper-base) with its tokenizer (whisper_base_tokenizer.json, 2480466 bytes, SHA-256 27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566, the tokenizer.json of openai/whisper-base), is redistributed unmodified.

Whisper by OpenAI. Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0''';

const _moonshineNotice = '''
The speech recognizer of Live camera built into this app, moonshine-tiny (moonshine_tiny_5s_f32.tflite, 109373140 bytes, SHA-256 16f281f1d3d23124e6adbdead8730d46f97cd105c299a9b94608d033c1151b12, Hugging Face: litert-community/moonshine-tiny) with its tokenizer (moonshine_tiny_tokenizer.json, 1985534 bytes, SHA-256 ed2324b3f699d8ba18a4030f33ba205d50b30ff46ff73f2b7d22661cc850efb4, from moonshine-ai/moonshine), is redistributed unmodified.

Moonshine by Useful Sensors (Moonshine AI). Licensed under the MIT License: https://opensource.org/license/mit''';

const _inflectNotice = '''
The speech synthesizer built into this app, Inflect-nano-v2 (inflect_text_encoder_fp16.tflite, 1782268 bytes, SHA-256 c778f516b8557c69734f4d4308084c411b8d3a4575566385d163209bf504f083; inflect_decoder_fp16.tflite, 6375948 bytes, SHA-256 155d5072f104248983015e82d1b90968ab2e87bd7f96c4016928fc83231efbd0; Hugging Face: litert-community/Inflect-Nano-v2), is licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

The four grapheme-to-phoneme files it reuses (config.json, g2p_dict.txt.gz, dp_g2p_matcha_fp16.tflite, g2p_meta.json; Hugging Face: litert-community/Matcha-TTS) are licensed under the MIT License: https://opensource.org/license/mit

All are redistributed unmodified; sizes and SHA-256 of every file are in assets/models/NOTICE.md.''';

const _catsNotice = '''
The photo built into this app for the self-test's detector check and the image chat, test_assets/cats.jpg (640×480 JPEG, SHA-256 dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e), is "Cats and remote controllers." by DocChewbacca on Flickr (https://www.flickr.com/photo.gne?id=210383891), as copied in the COCO 2017 validation set (image 39769, https://cocodataset.org).

It is licensed under the Creative Commons Attribution-ShareAlike 2.0 Generic licence (CC BY-SA 2.0): https://creativecommons.org/licenses/by-sa/2.0/''';

const _libjpegTurboNotice = '''
On Linux the network camera decodes JPEG frames with libjpeg-turbo's TurboJPEG library (lib/libturbojpeg.so.0 in the bundle, copied unmodified from the build machine's Debian/Ubuntu package libturbojpeg, or the system's copy).

This software is based in part on the work of the Independent JPEG Group.

libjpeg-turbo is covered by the IJG License (the libjpeg API library) and the Modified (3-clause) BSD License (the TurboJPEG API library): https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/LICENSE.md

Copyright (C) 2009-2023 D. R. Commander. All Rights Reserved.
Copyright (C) 2015 Viktor Szathmáry. All Rights Reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the libjpeg-turbo Project nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.''';
