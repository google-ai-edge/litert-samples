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

import 'package:file_selector/file_selector.dart';

import '../../../domain/ports/model_file_picker.dart';

/// [ModelFilePicker] over file_selector 1.1.0.
final class FileSelectorModelFilePicker implements ModelFilePicker {
  const FileSelectorModelFilePicker();

  @override
  ImportSupport get support {
    if (Platform.isAndroid) {
      // file_selector_android 0.5.2+11 reads a picked file into one Java
      // byte[] sized by an int (`toFileResponse`): a 2.5 GB .litertlm
      // overflows it and even a 180 MB file risks an out-of-memory.
      return const ImportUnsupported(
        'Importing files is not available on Android yet (the system picker '
        'plugin loads whole files into memory): adb push the .litertlm into '
        'a models folder, or use "Download from URL…".',
      );
    }
    if (Platform.isIOS) return const ImportFromFiles();
    return const ImportFromFolder();
  }

  @override
  Future<String?> pickModelFile() async {
    final group = Platform.isIOS
        ? const XTypeGroup(
            label: 'LiteRT-LM model',
            uniformTypeIdentifiers: ['public.data'],
          )
        : const XTypeGroup(label: 'LiteRT-LM model', extensions: ['litertlm']);
    final file = await openFile(
      acceptedTypeGroups: [group],
      confirmButtonText: 'Import',
    );
    return file?.path;
  }

  /// iOS: file_selector_ios opens the document picker `in: .import`, which
  /// copies each pick into `tmp/<bundle id>-Inbox/`: under
  /// `NSTemporaryDirectory()` (`Directory.systemTemp`), not under
  /// path_provider's `getTemporaryDirectory()` (the Caches folder).
  @override
  Future<String?> temporaryCopiesDirectory() async =>
      Platform.isIOS ? Directory.systemTemp.path : null;
}
