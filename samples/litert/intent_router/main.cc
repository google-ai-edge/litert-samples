/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Intent routing with a sentence embedder, for a tool-calling agent.
//
// This is the layer *before* a language model: given a free-text request and a set
// of candidate actions, decide which action applies, or decline. It exists because
// a small generator is a poor router. Measured on a Samsung SM-A145F (Exynos 3830,
// 3.5 GB RAM, Android 15), a 270M function-calling `.litertlm` bundle routed a
// paired prompt set at 22% with 72% refusals, taking 3120 ms per prompt and 983 MB
// peak RSS. A 22M-parameter sentence embedder routed the same set at 100% in 594 ms
// and 285 MB. The generator was failing at routing, not at generation, so spending
// 3 seconds and a gigabyte on the routing decision was the wrong trade.
//
// Three properties of the scoring below are the transferable part:
//
//   1. PROTOTYPES, not centroids. Each action is represented by several
//      prototypes and scored by max cosine over them, not by a single embedded
//      description. One centroid per action cannot represent that users phrase
//      requests many ways; on a paired prompt set this is the single largest
//      accuracy lever available.
//
//   2. `clarify` IS A CANDIDATE. Out-of-domain input has no correct action, and a
//      plain argmax over actions will always invent one. Declining competes in the
//      same comparison, so "what is the weather" can lose to nothing.
//
//   3. THE MARGIN IS THE OUTPUT THAT MATTERS. Print the gap between the winner and
//      the best alternative. Callers use it to decide whether to act or ask, which
//      is the decision an agent actually has to make.
//
// The one thing this cannot do is distinguish antonyms. "switch the torch on" and
// "switch the torch off" sit at cosine 0.949 -- embeddings encode topic, not
// polarity. If your action set contains an on/off pair, resolve the polarity with a
// lexical check and use the embedding only to select the action class. See the
// README.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "sentencepiece_processor.h"  // from @sentencepiece

ABSL_FLAG(std::string, tokenizer, "", "Path to the SentencePiece model file.");
ABSL_FLAG(std::string, embedder, "",
          "Path to the sentence embedder .tflite.");
ABSL_FLAG(int, sequence_length, 128,
          "Padded sequence length. Must match the embedder's input shape.");
ABSL_FLAG(int, num_threads, 4, "CPU threads.");
ABSL_FLAG(std::string, accelerator, "cpu",
          "Comma delimited accelerators: cpu, gpu. Falls back to CPU.");
ABSL_FLAG(double, abstain_margin, 0.02,
          "Decline when the winner is within this cosine margin of the best "
          "alternative. Calibrate on real traces; 0.02 was measured on one "
          "device with one action set and is a starting point, not a constant.");
ABSL_FLAG(std::string, utterances, "",
          "Semicolon-separated utterances to route. Overrides the built-in "
          "evaluation set.");

namespace litert {
namespace {

// ---------------------------------------------------------------------------
// Action set. Four actions with three properties worth noticing:
//
//   - open/close flashlight is an ANTONYM PAIR. This is the case a bi-encoder
//     cannot resolve, and it is included on purpose so the limitation is
//     reproducible rather than theoretical.
//   - query_calendar has many natural phrasings ("my agenda", "am i free"),
//     which is why its prototype list is long.
//   - `no_action` is the decline class. See ABOVE on why it must exist.
// ---------------------------------------------------------------------------

struct Action {
  std::string name;
  std::vector<std::string> prototypes;
};

const std::vector<Action>& Actions() {
  static const std::vector<Action>* const actions = new std::vector<Action>{
      {"open_flashlight",
       {"Turns the phone's flashlight on.",
        "switch the flashlight on",
        "turn on the light",
        "it is dark in here",
        "lights on please"}},
      {"close_flashlight",
       {"Turns the phone's flashlight off.",
        "switch the flashlight off",
        "turn off the light",
        "kill the torch",
        "that beam is giving me a headache"}},
      {"query_calendar",
       {"Lists the events on the user's calendar for today.",
        "what is on my calendar today",
        "what's my schedule",
        "my agenda",
        "do i have meetings today",
        "am i free today"}},
      {"no_action",
       {"Ask about something the phone cannot do.",
        "what is the weather",
        "who won the match",
        "play some music",
        "how do i tie a knot"}},
  };
  return *actions;
}

// Built-in evaluation set. Each intent appears once with a wording a user would
// plausibly type and once that avoids the action's own vocabulary. A router that
// matches keywords rather than meaning scores well on the first and badly on the
// second, so the pairing is what makes the number mean anything.
struct Probe {
  const char* utterance;
  const char* expected;  // "no_action" means: must decline.
};

const std::vector<Probe>& Probes() {
  static const std::vector<Probe>* const probes =
      new std::vector<Probe>{
          {"turn on the flashlight", "open_flashlight"},
          {"it is dark in here", "open_flashlight"},
          {"switch the flashlight on", "open_flashlight"},
          {"turn off the flashlight", "close_flashlight"},
          {"kill the torch", "close_flashlight"},
          {"that beam is giving me a headache", "close_flashlight"},
          {"what is on my calendar today", "query_calendar"},
          {"do i have meetings today", "query_calendar"},
          {"am i free today", "query_calendar"},
          {"my agenda", "query_calendar"},
          {"what is the weather in nairobi", "no_action"},
          {"play some music", "no_action"},
      };
  return *probes;
}

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

float Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  ABSL_QCHECK(!a.empty() && a.size() == b.size());
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const double mag = std::sqrt(na) * std::sqrt(nb);
  return mag == 0.0 ? 0.0f : static_cast<float>(dot / mag);
}

// L2-normalize in place, so later comparisons are plain dot products and the
// argmax is invariant to vector length.
void Normalize(std::vector<float>* v) {
  double sum = 0.0;
  for (float x : *v) sum += x * x;
  const float norm = std::sqrt(sum);
  if (norm > 0.0f) {
    for (float& x : *v) x /= norm;
  }
}

void Tokenize(sentencepiece::SentencePieceProcessor* sp,
              const std::string& text, int seq_len, std::vector<int>* ids,
              std::vector<int>* mask) {
  std::vector<int> body;
  ABSL_CHECK_OK(sp->Encode(text, &body));
  if (static_cast<int>(body.size()) > seq_len - 2) body.resize(seq_len - 2);
  body.insert(body.begin(), sp->bos_id());
  body.push_back(sp->eos_id());

  ids->assign(seq_len, sp->pad_id());
  mask->assign(seq_len, 0);
  for (size_t i = 0; i < body.size(); ++i) {
    (*ids)[i] = body[i];
    (*mask)[i] = 1;
  }
}

absl::StatusOr<std::vector<float>> Embed(CompiledModel* model,
                                         std::vector<TensorBuffer>* in,
                                         std::vector<TensorBuffer>* out,
                                         const std::vector<int>& ids,
                                         const std::vector<int>& mask) {
  LITERT_RETURN_IF_ERROR((*in)[0].Write<int>(ids));
  if (in->size() > 1) LITERT_RETURN_IF_ERROR((*in)[1].Write<int>(mask));
  LITERT_RETURN_IF_ERROR(model->Run(*in, *out));
  LITERT_ASSIGN_OR_RETURN(size_t bytes, (*out)[0].PackedSize());
  std::vector<float> vec(bytes / sizeof(float));
  LITERT_RETURN_IF_ERROR((*out)[0].Read(absl::MakeSpan(vec)));
  // The embedder graph already mean-pools; only normalize here.
  Normalize(&vec);
  return vec;
}

// One prototype per action, embedded once at startup. Startup cost scales with
// the number of prototypes: on the device referenced in the README, a single
// embedding pass took ~600 ms, so a large prototype bank dominates cold start.
// Embed the actions the caller needs first and defer the rest, or keep the bank
// small -- the thing that needs testing is whether that tradeoff holds.
std::vector<std::vector<std::vector<float>>> BuildPrototypes(
    CompiledModel* model, std::vector<TensorBuffer>* in,
    std::vector<TensorBuffer>* out,
    sentencepiece::SentencePieceProcessor* sp, int seq_len) {
  std::vector<std::vector<std::vector<float>>> protos;
  protos.reserve(Actions().size());
  for (const Action& action : Actions()) {
    std::vector<std::vector<float>> group;
    for (const std::string& text : action.prototypes) {
      std::vector<int> ids, mask;
      Tokenize(sp, text, seq_len, &ids, &mask);
      auto vec = Embed(model, in, out, ids, mask);
      ABSL_CHECK_OK(vec.status());
      group.push_back(*vec);
    }
    protos.push_back(std::move(group));
  }
  return protos;
}

struct Decision {
  std::string action;
  double margin;
  bool declined;
  std::string runner_up;
};

// The scoring. Note that the runner-up is built by EXCLUDING the winner from the
// candidate list; folding the winner's own score into the comparison pins the
// margin at zero, which looks like a working threshold and is not one.
Decision Route(const std::vector<float>& query,
               const std::vector<std::vector<std::vector<float>>>& protos,
               double abstain_margin) {
  std::vector<std::pair<std::string, float>> scores;
  for (size_t i = 0; i < Actions().size(); ++i) {
    float best = -1.0f;
    for (const std::vector<float>& proto : protos[i]) {
      best = std::max(best, Cosine(query, proto));
    }
    scores.emplace_back(Actions()[i].name, best);
  }
  std::sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  const std::string& winner = scores[0].first;
  const float runner_up = scores[1].second;
  const double margin = scores[0].second - runner_up;

  if (margin < abstain_margin) {
    return {winner, margin, /*declined=*/true, scores[1].first};
  }
  return {winner, margin, /*declined=*/false, scores[1].first};
}

absl::Status RealMain() {
  ABSL_QCHECK(!absl::GetFlag(FLAGS_tokenizer).empty()) << "--tokenizer required";
  ABSL_QCHECK(!absl::GetFlag(FLAGS_embedder).empty()) << "--embedder required";
  const int seq_len = absl::GetFlag(FLAGS_sequence_length);
  const double abstain_margin = absl::GetFlag(FLAGS_abstain_margin);

  sentencepiece::SentencePieceProcessor sp;
  if (auto load_status = sp.Load(absl::GetFlag(FLAGS_tokenizer)); !load_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to load tokenizer model: ", load_status.ToString()));
  }

  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  if (absl::StrContains(absl::AsciiStrToLower(absl::GetFlag(FLAGS_accelerator)),
                        "gpu")) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu, options.GetGpuOptions());
    gpu.SetPrecision(GpuOptions::Precision::kFp32);
    options.SetHardwareAccelerators(HwAccelerators::kGpu | HwAccelerators::kCpu);
  } else {
    LITERT_ASSIGN_OR_RETURN(auto& cpu, options.GetCpuOptions());
    cpu.SetNumThreads(absl::GetFlag(FLAGS_num_threads));
    options.SetHardwareAccelerators(HwAccelerators::kCpu);
  }

  // EnvironmentOptions rather than the deprecated Environment::Option. An empty
  // option list: the dispatch directory is only needed for NPU, which this
  // sample does not select.
  const std::vector<EnvironmentOptions::Option> no_env_options;
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create(EnvironmentOptions(
                                                  no_env_options)));
  LITERT_ASSIGN_OR_RETURN(auto model,
                          CompiledModel::Create(env, absl::GetFlag(FLAGS_embedder),
                                                options));
  LITERT_ASSIGN_OR_RETURN(auto in, model.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto out, model.CreateOutputBuffers());

  const auto protos = BuildPrototypes(&model, &in, &out, &sp, seq_len);
  LOG(INFO) << "embedded " << Actions().size() << " action prototypes";

  std::vector<std::string> utterances;
  if (!absl::GetFlag(FLAGS_utterances).empty()) {
    for (absl::string_view piece :
         absl::StrSplit(absl::GetFlag(FLAGS_utterances), ';', absl::SkipEmpty())) {
      utterances.emplace_back(piece);
    }
    for (const std::string& text : utterances) {
      std::vector<int> ids, mask;
      Tokenize(&sp, text, seq_len, &ids, &mask);
      auto vec = Embed(&model, &in, &out, ids, mask);
      LITERT_RETURN_IF_ERROR(vec.status());
      const Decision d = Route(*vec, protos, abstain_margin);
      std::cout << absl::StrFormat("%s  ->  %s  margin=%.3f%s\n", text,
                                   d.action, d.margin,
                                   d.declined ? "  (declined)" : "");
    }
    return absl::OkStatus();
  }

  // Evaluation mode. Scoring only over probes whose expected action was offered,
  // so a narrow action set is not credited for prompts it had no way to answer.
  int correct = 0;
  int declined = 0;
  for (const Probe& probe : Probes()) {
    std::vector<int> ids, mask;
    Tokenize(&sp, probe.utterance, seq_len, &ids, &mask);
    auto vec = Embed(&model, &in, &out, ids, mask);
    LITERT_RETURN_IF_ERROR(vec.status());
    const Decision d = Route(*vec, protos, abstain_margin);

    const std::string expected = probe.expected;
    const bool should_decline = expected == "no_action";
    const bool ok =
        should_decline ? d.declined : (!d.declined && d.action == expected);
    if (ok) ++correct;
    if (d.declined) ++declined;

    std::cout << absl::StrFormat(
        "%s %-40s want=%-16s got=%-16s margin=%.3f%s\n",
        ok ? "ok  " : "MISS", probe.utterance, expected, d.action, d.margin,
        d.declined ? " declined" : "");
  }
  const int total = static_cast<int>(Probes().size());
  std::cout << absl::StrCat("\naccuracy ", correct, "/", total, "  declined ",
                            declined, "\n");
  std::cout << "A high accuracy here means the pairing is too easy. Add "
               "paraphrases that avoid each action's vocabulary, plus "
               "out-of-domain and negated utterances, before trusting it.\n";
  return absl::OkStatus();
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = litert::RealMain();
  if (!status.ok()) {
    LOG(ERROR) << "intent_router failed: " << status;
    return 1;
  }
  return 0;
}