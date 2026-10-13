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
import 'package:litert_edge_demos/data/services/settings/settings_store.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:shared_preferences_platform_interface/in_memory_shared_preferences_async.dart';
import 'package:shared_preferences_platform_interface/shared_preferences_async_platform_interface.dart';
import 'package:shared_preferences_platform_interface/types.dart';

import '../../../fakes/fake_settings_store.dart';

void main() {
  test('reads, writes and clears typed settings', () async {
    final store = InMemorySettingsStore();
    final settings = TypedSettings(store: store);
    const flag = BoolSetting('test.flag');
    const url = Settings.networkCameraUrl;

    expect((await settings.read(url) as Ok<String?>).value, isNull);
    expect(
      await settings.write(url, 'http://192.168.1.23:8080/video'),
      isA<Ok<void>>(),
    );
    expect(await settings.write(flag, true), isA<Ok<void>>());
    expect(
      (await settings.read(url) as Ok<String?>).value,
      'http://192.168.1.23:8080/video',
    );
    expect((await settings.read(flag) as Ok<bool?>).value, isTrue);

    expect(await settings.clear(url), isA<Ok<void>>());
    expect((await settings.read(url) as Ok<String?>).value, isNull);
  });

  test('a storage failure is an error, never a silent default', () async {
    final store = InMemorySettingsStore()..failWith = Exception('disk');
    final settings = TypedSettings(store: store);

    expect(await settings.read(Settings.chatModel), isA<Error<String?>>());
    expect(await settings.write(Settings.chatModel, 'x'), isA<Error<void>>());
  });

  group('writePair: never a half-written pair', () {
    const first = StringSetting('pair.first');
    const second = StringSetting('pair.second');

    test('writes both, first then second', () async {
      final store = InMemorySettingsStore();
      final settings = TypedSettings(store: store);

      expect(
        await settings.writePair((first, 'a'), (second, 'b')),
        isA<Ok<void>>(),
      );
      expect(store.values, {'pair.first': 'a', 'pair.second': 'b'});
    });

    test('the second fails: the first is put back as it was', () async {
      final store = InMemorySettingsStore()
        ..values.addAll({'pair.first': 'old', 'pair.second': 'old2'})
        ..failWritesOf.add('pair.second');
      final settings = TypedSettings(store: store);

      expect(
        await settings.writePair((first, 'new'), (second, 'new2')),
        isA<Error<void>>(),
      );
      expect(store.values, {'pair.first': 'old', 'pair.second': 'old2'});
    });

    test(
      'the second fails and the first was unset: it is unset again',
      () async {
        final store = InMemorySettingsStore()..failWritesOf.add('pair.second');
        final settings = TypedSettings(store: store);

        expect(
          await settings.writePair((first, 'new'), (second, 'new2')),
          isA<Error<void>>(),
        );
        expect(store.values, isEmpty);
      },
    );

    test('the first fails: nothing is written', () async {
      final store = InMemorySettingsStore()..failWritesOf.add('pair.first');
      final settings = TypedSettings(store: store);

      expect(
        await settings.writePair((first, 'new'), (second, 'new2')),
        isA<Error<void>>(),
      );
      expect(store.values, isEmpty);
    });
  });

  test('the shared-preferences store prefixes every key with app.', () async {
    final platform = InMemorySharedPreferencesAsync.empty();
    SharedPreferencesAsyncPlatform.instance = platform;
    final store = SharedPreferencesSettingsStore();

    await store.setString('chat.model', 'custom');

    final settings = TypedSettings(store: store);
    expect(
      (await settings.read(Settings.chatModel) as Ok<String?>).value,
      'custom',
    );
    expect(
      await platform.getString(
        'app.chat.model',
        const SharedPreferencesOptions(),
      ),
      'custom',
    );
  });
}
