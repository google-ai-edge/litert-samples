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

import 'package:litert_edge_demos/config/model_catalog.dart'
    show kMaxSkillTokens;
import 'package:litert_edge_demos/data/services/skills/skill_store_service.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';

/// [SkillStoreService] without a disk: seeding does nothing and every scan
/// finds [catalog] (by default an empty skills folder). Wrap it in a
/// `SkillRepository` for a chat opened without skills.
final class FakeSkillStore implements SkillStoreService {
  FakeSkillStore({this.catalog = emptyCatalog});

  static const emptyCatalog = SkillCatalog(
    directory: '/fake/skills',
    fingerprint: 'empty',
  );

  /// What every scan returns.
  SkillCatalog catalog;

  @override
  int get maxBytes => 4 * 1024;

  @override
  int get maxTokens => kMaxSkillTokens;

  @override
  Future<SeedReport> seed() async =>
      const SeedReport(created: [], updated: [], keptEdits: [], skipped: true);

  @override
  Future<SkillCatalog> scan() async => catalog;
}
