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

// A port: an interface that the domain and the view models depend on and an
// outer layer implements. Every port lives in lib/domain/ports/;
// function-typed dependencies stay next to their one consumer.

import '../../utils/result.dart';
import '../models/knowledge.dart';
import '../models/knowledge_query.dart';

/// What Demo 1's turn responder needs from the knowledge base: one gated
/// search per question, typed or spoken. `KnowledgeRepository` implements
/// it; responder tests pass a fake.
abstract interface class KnowledgeRetriever {
  /// Ok with [RetrievalOutcome.used] (excerpts for the prompt),
  /// [RetrievalOutcome.belowGate] or [RetrievalOutcome.unavailable] (with
  /// the reason), gated at the [query]'s kind's similarity; Error when the
  /// search itself failed, a [KnowledgeQueryTooLongException] included (the
  /// question was not searched, nothing was cut). Never throws.
  Future<Result<Retrieval>> retrieve(KnowledgeQuery query);
}
