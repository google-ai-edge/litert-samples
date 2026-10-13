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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/skill_repository.dart';
import 'package:litert_edge_demos/data/services/skills/skill_store_service.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';

import '../../../../integration_test/support/skill_fixtures.dart';

final class FixedBundle implements BundledSkillSource {
  FixedBundle(this.skills);

  Map<String, String> skills;

  @override
  Future<Map<String, String>> load() async => Map.of(skills);
}

String bundledText(String name) =>
    File('assets/skills/$name/SKILL.md').readAsStringSync();

/// The user-writable skills folder.
void main() {
  late Directory root;
  late Directory skillsDir;
  late FixedBundle bundle;
  late SkillStoreService store;

  File file(String relative) => File('${skillsDir.path}/$relative');

  void write(String relative, String text) {
    file(relative)
      ..parent.createSync(recursive: true)
      ..writeAsStringSync(text);
  }

  setUp(() {
    root = Directory.systemTemp.createTempSync('skill_store_test');
    skillsDir = Directory('${root.path}/skills');
    bundle = FixedBundle({
      'current-time': bundledText('current-time'),
      'device-info': bundledText('device-info'),
    });
    store = SkillStoreService(
      directory: () async => skillsDir,
      bundled: bundle,
    );
  });

  tearDown(() => root.deleteSync(recursive: true));

  group('seed', () {
    test('copies each bundled skill whose folder is absent', () async {
      final report = await store.seed();

      expect(report.created, unorderedEquals(['current-time', 'device-info']));
      expect(
        file('current-time/SKILL.md').readAsStringSync(),
        bundledText('current-time'),
      );
      expect(file('.seed-version').existsSync(), isTrue);
    });

    test('keeps a skill the user edited, also across an app update', () async {
      await store.seed();
      write(
        'current-time/SKILL.md',
        '${bundledText('current-time')}\nMy own note.',
      );
      bundle.skills['current-time'] = '${bundledText('current-time')}\nv2';

      final report = await store.seed();

      expect(report.keptEdits, ['current-time']);
      expect(
        file('current-time/SKILL.md').readAsStringSync(),
        endsWith('My own note.'),
      );
    });

    test(
      'replaces an unedited bundled skill when the bundle changed',
      () async {
        await store.seed();
        bundle.skills['current-time'] = '${bundledText('current-time')}\nv2';

        final report = await store.seed();

        expect(report.updated, ['current-time']);
        expect(
          file('current-time/SKILL.md').readAsStringSync(),
          endsWith('v2'),
        );
      },
    );

    test('a bundled skill the user deleted stays deleted: the same bundle is '
        'not seeded again', () async {
      await store.seed();
      Directory('${skillsDir.path}/current-time').deleteSync(recursive: true);

      final report = await store.seed();

      expect(report.skipped, isTrue);
      expect(report.created, isEmpty);
      expect(Directory('${skillsDir.path}/current-time').existsSync(), isFalse);
    });

    test('an app update re-seeds: new bundled skills appear, a deleted one '
        'stays deleted', () async {
      await store.seed();
      Directory('${skillsDir.path}/current-time').deleteSync(recursive: true);
      bundle.skills
        ..['current-time'] = '${bundledText('current-time')}\nv2'
        ..['kid-clock'] = kidClockSkillMd;

      final report = await store.seed();

      expect(report.skipped, isFalse);
      expect(report.created, ['kid-clock']);
      expect(report.stayedDeleted, ['current-time']);
      expect(Directory('${skillsDir.path}/current-time').existsSync(), isFalse);
      expect(file('kid-clock/SKILL.md').existsSync(), isTrue);
    });
  });

  group('retiring skills an earlier version bundled', () {
    test('an unedited copy is deleted with its folder; an edited one is '
        'kept; the record forgets the retired one', () async {
      bundle.skills['timer'] = '---\nname: timer\ndescription: Old.\n---\nx';
      bundle.skills['camera-watch'] =
          '---\nname: camera-watch\ndescription: Old.\n---\ny';
      await store.seed();
      write(
        'camera-watch/SKILL.md',
        '---\nname: camera-watch\ndescription: Mine.\n---\ny',
      );
      bundle.skills
        ..remove('timer')
        ..remove('camera-watch');

      final report = await store.seed();

      expect(report.retired, ['timer']);
      expect(Directory('${skillsDir.path}/timer').existsSync(), isFalse);
      expect(report.keptEdits, ['camera-watch']);
      expect(file('camera-watch/SKILL.md').existsSync(), isTrue);
      // A later bundle with a "timer" again is a new skill, created.
      bundle.skills['timer'] = '---\nname: timer\ndescription: New.\n---\nz';
      expect((await store.seed()).created, ['timer']);
    });

    test(
      'a retired skill\'s folder with the user\'s own files in it stays',
      () async {
        bundle.skills['timer'] = '---\nname: timer\ndescription: Old.\n---\nx';
        await store.seed();
        write('timer/notes.txt', 'mine');
        bundle.skills.remove('timer');

        final report = await store.seed();

        expect(report.retired, ['timer']);
        expect(file('timer/SKILL.md').existsSync(), isFalse);
        expect(file('timer/notes.txt').existsSync(), isTrue);
      },
    );
  });

  group('scan', () {
    test('the seeded skills load; folder/SKILL.md and top-level .md both '
        'count; hidden files are skipped', () async {
      await store.seed();
      write('kid-clock/SKILL.md', kidClockSkillMd);
      write(
        'notes.md',
        '---\nname: notes\ndescription: Plain notes.\n---\nBe nice.',
      );
      write('.hidden/SKILL.md', 'garbage');

      final catalog = await store.scan();

      expect(catalog.directory, skillsDir.path);
      expect(catalog.errors, isEmpty);
      expect(catalog.skills.map((s) => s.path), [
        'current-time/SKILL.md',
        'device-info/SKILL.md',
        'kid-clock/SKILL.md',
        'notes.md',
      ]);
      expect(catalog.agentSkills.map((s) => s.name), [
        'current-time',
        'device-info',
        'kid-clock',
        'notes',
      ]);
    });

    test('every bad file is listed with why, and the good ones still load', () async {
      write('current-time/SKILL.md', bundledText('current-time'));
      write('broken/SKILL.md', brokenSkillMd);
      write('numeric/SKILL.md', '---\nname: 123\ndescription: x\n---\nBody');
      file('binary/SKILL.md')
        ..parent.createSync(recursive: true)
        ..writeAsBytesSync([0x2d, 0x2d, 0x2d, 0xff, 0xfe, 0x0a]);
      write(
        'huge/SKILL.md',
        '---\nname: huge\ndescription: x\n---\n${'a' * 40000}',
      );
      write(
        'zz-dupe/SKILL.md',
        '---\nname: Current-Time\ndescription: Again.\n---\nHi.',
      );
      write(
        'typo/SKILL.md',
        '---\nname: typo\ndescription: x\n---\nCall the `run_intent` tool with '
            'intent `current_tme`.',
      );
      write(
        'vague/SKILL.md',
        '---\nname: vague\ndescription: x\n---\nCall the `run_intent` tool.',
      );
      write(
        'js/SKILL.md',
        '---\nname: js\ndescription: x\n---\nCall the `run_js` tool.',
      );
      Directory('${skillsDir.path}/empty').createSync(recursive: true);

      final catalog = await store.scan();

      expect(catalog.agentSkills.map((s) => s.name), ['current-time']);
      String why(String path) =>
          catalog.errors.singleWhere((e) => e.path == path).message;
      expect(why('broken/SKILL.md'), contains("expected a '---' fenced"));
      expect(why('numeric/SKILL.md'), contains('quote name and description'));
      expect(why('binary/SKILL.md'), 'Not valid UTF-8 text');
      expect(why('huge/SKILL.md'), contains('Larger than 4 KB'));
      expect(
        why('zz-dupe/SKILL.md'),
        contains('Duplicate name "Current-Time"'),
      );
      expect(why('zz-dupe/SKILL.md'), contains('current-time/SKILL.md'));
      expect(why('typo/SKILL.md'), contains('Unknown intent `current_tme`'));
      expect(why('vague/SKILL.md'), contains('names no intent'));
      expect(why('js/SKILL.md'), contains('run_js'));
      expect(why('empty/'), 'This folder has no SKILL.md');
    });

    test('a skill must stay small: over 4 KB is refused, and so is one over '
        '600 tokens of instructions, which would sit in every turn\'s '
        'budget', () async {
      write('current-time/SKILL.md', bundledText('current-time'));
      // ~5 KB: over the file limit.
      write(
        'big/SKILL.md',
        '---\nname: big\ndescription: x\n---\n${'word ' * 1000}',
      );
      // ~3 KB: under the file limit, but about 1000 tokens.
      write(
        'wordy/SKILL.md',
        '---\nname: wordy\ndescription: Says a lot.\n---\nCall the '
            '`run_intent` tool with intent `current_time`. ${'Be precise. ' * 240}',
      );

      final catalog = await store.scan();

      expect(store.maxBytes, 4 * 1024);
      expect(catalog.agentSkills.map((s) => s.name), ['current-time']);
      String why(String path) =>
          catalog.errors.singleWhere((e) => e.path == path).message;
      expect(why('big/SKILL.md'), contains('Larger than 4 KB'));
      expect(
        why('wordy/SKILL.md'),
        allOf(contains('tokens'), contains('at most $kMaxSkillTokens')),
      );
    });

    test('an oversized file is refused by its length, without reading '
        'it', () async {
      write('big/SKILL.md', 'x' * (6 * 1024));
      // Unreadable: only a read fails; a length check does not.
      Process.runSync('chmod', ['000', file('big/SKILL.md').path]);
      addTearDown(
        () => Process.runSync('chmod', ['644', file('big/SKILL.md').path]),
      );

      final catalog = await store.scan();

      expect(catalog.errors.single.message, 'Larger than 4 KB (6144 bytes)');
    });

    test('the bundled skills are well inside the token limit', () async {
      for (final name in ['current-time', 'device-info']) {
        expect(
          estimateSkillTokens(bundledText(name)),
          lessThan(kMaxSkillTokens ~/ 2),
          reason: name,
        );
      }
    });

    test('the fingerprint changes with any file and only then', () async {
      await store.seed();
      final first = await store.scan();
      final same = await store.scan();
      write('kid-clock/SKILL.md', kidClockSkillMd);
      final added = await store.scan();
      write('kid-clock/SKILL.md', '$kidClockSkillMd\n');
      final edited = await store.scan();

      expect(same.fingerprint, first.fingerprint);
      expect(added.fingerprint, isNot(first.fingerprint));
      expect(edited.fingerprint, isNot(added.fingerprint));
    });

    test(
      'an unusable directory is a catalog with the error, not a throw',
      () async {
        final broken = SkillStoreService(
          directory: () async => throw const SkillStoreException(
            'External storage is not available',
          ),
          bundled: bundle,
        );

        final catalog = await broken.scan();

        expect(catalog.skills, isEmpty);
        expect(catalog.directory, isNull);
        expect(catalog.storeError, contains('External storage'));
      },
    );
  });

  group('SkillRepository', () {
    test(
      'seeds once, then rescans; the catalog changes only with the files',
      () async {
        final repo = SkillRepository(store: store);
        addTearDown(repo.dispose);
        final notified = <SkillCatalog?>[];
        repo.catalog.addListener(() => notified.add(repo.catalog.value));

        final first = await repo.ensureLoaded();
        expect(first.agentSkills.map((s) => s.name), [
          'current-time',
          'device-info',
        ]);
        expect(await repo.ensureLoaded(), same(first), reason: 'no rescan');
        expect(await repo.refresh(), same(first), reason: 'unchanged');
        expect(notified, hasLength(1));

        write('kid-clock/SKILL.md', kidClockSkillMd);
        final second = await repo.refresh();

        expect(second.agentSkills.map((s) => s.name), contains('kid-clock'));
        expect(notified, [first, second]);
      },
    );

    test('concurrent refreshes share one scan', () async {
      final repo = SkillRepository(store: store);
      addTearDown(repo.dispose);

      final results = await Future.wait([repo.refresh(), repo.refresh()]);

      expect(results.first, same(results.last));
    });
  });
}
