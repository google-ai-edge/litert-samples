// Video-stack weight loading. See sam2v_weights.h.

#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_weights.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::sam2_video {

namespace {

// The gemma3 loader's Gemma-norm heuristic: +1.0 on any tensor whose name
// contains "layernorm" or ends with "norm.weight". Three video keys match
// (memory_attention.norm.weight, memory_encoder.fuser.{0,1}.norm.weight);
// their loaded values are compensated back below.
bool LoaderAddsGemmaOffset(const std::string& name) {
  return absl::StrContains(name, "layernorm") ||
         absl::EndsWith(name, "norm.weight");
}

}  // namespace

std::vector<WeightSpec> GetVideoWeightSpecs(const Sam2VideoConfig& config) {
  std::vector<WeightSpec> specs;
  auto add = [&](const std::string& name, const std::vector<int>& shape,
                 float scale = 0.02f) {
    specs.push_back({name, shape, scale});
  };
  const int hd = config.hidden;   // 256
  const int mc = config.mem_ch;   // 64
  const int hw = config.hw();     // 4096 at 1024

  for (int l = 0; l < config.ma_layers; ++l) {
    std::string p = absl::StrCat("memory_attention.layers.", l);
    for (const std::string& attn :
         {std::string(".self_attn"), std::string(".cross_attn_image")}) {
      const int kv_in = attn == ".self_attn" ? hd : mc;
      add(p + attn + ".q_proj.weight", {hd, hd});
      add(p + attn + ".q_proj.bias", {hd}, 0.0f);
      add(p + attn + ".k_proj.weight", {hd, kv_in});
      add(p + attn + ".k_proj.bias", {hd}, 0.0f);
      add(p + attn + ".v_proj.weight", {hd, kv_in});
      add(p + attn + ".v_proj.bias", {hd}, 0.0f);
      add(p + attn + ".out_proj.weight", {hd, hd});
      add(p + attn + ".out_proj.bias", {hd}, 0.0f);
    }
    add(p + ".linear1.weight", {config.ff_hidden, hd});
    add(p + ".linear1.bias", {config.ff_hidden}, 0.0f);
    add(p + ".linear2.weight", {hd, config.ff_hidden});
    add(p + ".linear2.bias", {hd}, 0.0f);
    for (int n = 1; n <= 3; ++n) {
      add(absl::StrCat(p, ".norm", n, ".weight"), {hd}, 1.0f);
      add(absl::StrCat(p, ".norm", n, ".bias"), {hd}, 0.0f);
    }
  }
  add("memory_attention.norm.weight", {hd}, 1.0f);
  add("memory_attention.norm.bias", {hd}, 0.0f);

  // Memory encoder.
  const int ds_ch[5] = {1, 4, 16, 64, 256};
  for (int i = 0; i < 4; ++i) {
    add(absl::StrCat("memory_encoder.mask_downsampler.conv", i, ".weight"),
        {ds_ch[i + 1], 3, 3, ds_ch[i]});
    add(absl::StrCat("memory_encoder.mask_downsampler.conv", i, ".bias"),
        {ds_ch[i + 1]}, 0.0f);
    add(absl::StrCat("memory_encoder.mask_downsampler.norm", i, ".weight"),
        {ds_ch[i + 1]}, 1.0f);
    add(absl::StrCat("memory_encoder.mask_downsampler.norm", i, ".bias"),
        {ds_ch[i + 1]}, 0.0f);
  }
  add("memory_encoder.mask_downsampler.conv4.weight", {hd, 1, 1, hd});
  add("memory_encoder.mask_downsampler.conv4.bias", {hd}, 0.0f);
  add("memory_encoder.pix_feat_proj.weight", {hd, 1, 1, hd});
  add("memory_encoder.pix_feat_proj.bias", {hd}, 0.0f);
  for (int i = 0; i < 2; ++i) {
    std::string p = absl::StrCat("memory_encoder.fuser.", i);
    add(p + ".dwconv.weight", {hd, 7, 7, 1});
    add(p + ".dwconv.bias", {hd}, 0.0f);
    add(p + ".norm.weight", {hd}, 1.0f);
    add(p + ".norm.bias", {hd}, 0.0f);
    add(p + ".pwconv1.weight", {4 * hd, hd});
    add(p + ".pwconv1.bias", {4 * hd}, 0.0f);
    add(p + ".pwconv2.weight", {hd, 4 * hd});
    add(p + ".pwconv2.bias", {hd}, 0.0f);
    add(p + ".gamma", {hd}, 1.0f);
  }
  add("memory_encoder.out_proj.weight", {mc, 1, 1, hd});
  add("memory_encoder.out_proj.bias", {mc}, 0.0f);

  // Pointer heads and per-frame constants.
  add("maskmem_tpos_enc", {7, 1, 1, mc});
  add("no_obj_ptr", {1, hd});
  add("no_obj_embed_spatial", {1, mc});
  for (int i = 0; i < 3; ++i) {
    add(absl::StrCat("obj_ptr_proj.layers.", i, ".weight"), {hd, hd});
    add(absl::StrCat("obj_ptr_proj.layers.", i, ".bias"), {hd}, 0.0f);
  }
  add("obj_ptr_tpos_proj.weight", {mc, hd});
  add("obj_ptr_tpos_proj.bias", {mc}, 0.0f);

  // The video decoder emits all four mask tokens; the image spec only
  // carries hypernetworks 1..3.
  add("sam_mask_decoder.output_hypernetworks_mlps.0.layers.0.weight",
      {hd, hd});
  add("sam_mask_decoder.output_hypernetworks_mlps.0.layers.0.bias", {hd},
      0.0f);
  add("sam_mask_decoder.output_hypernetworks_mlps.0.layers.1.weight",
      {hd, hd});
  add("sam_mask_decoder.output_hypernetworks_mlps.0.layers.1.bias", {hd},
      0.0f);
  add("sam_mask_decoder.output_hypernetworks_mlps.0.layers.2.weight",
      {32, hd});
  add("sam_mask_decoder.output_hypernetworks_mlps.0.layers.2.bias", {32},
      0.0f);

  // Derived tables (export-time constants).
  add("tables.rope_cos", {hw, hd}, 1.0f);
  add("tables.rope_sin", {hw, hd}, 0.02f);
  add("tables.vision_pos_scaled", {hw, hd}, 0.02f);
  add("tables.mem_pos", {hw, mc}, 0.02f);
  add("tables.track_sparse", {2, hd}, 0.02f);
  return specs;
}

absl::Status LoadVideoWeights(const Sam2VideoConfig& config,
                              const std::string& path, WeightMap& weights) {
  auto loader_or = SafetensorLoader::Load(path);
  if (!loader_or.ok()) return loader_or.status();
  auto loader = std::move(*loader_or);

  for (const WeightSpec& spec : GetVideoWeightSpecs(config)) {
    auto handle_or = loader.LoadTensor(
        spec.name, SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32);
    if (!handle_or.ok()) {
      return absl::NotFoundError(absl::StrCat("checkpoint missing ", spec.name,
                                              ": ",
                                              handle_or.status().message()));
    }
    TfTensor tensor(*handle_or);
    const auto& loaded_shape = tensor.GetShape();
    std::vector<int> loaded(loaded_shape.begin(), loaded_shape.end());
    if (loaded != spec.shape) {
      return absl::FailedPreconditionError(absl::StrCat(
          spec.name, ": checkpoint shape [", absl::StrJoin(loaded, ","),
          "] != expected [", absl::StrJoin(spec.shape, ","), "]"));
    }
    if (LoaderAddsGemmaOffset(spec.name)) {
      auto buffer = tensor.GetBuffer();
      if (!buffer.ok()) return buffer.status();
      auto lock = buffer->LockMutable();
      float* p = reinterpret_cast<float*>(lock.data());
      size_t count = lock.size() / sizeof(float);
      for (size_t i = 0; i < count; ++i) p[i] -= 1.0f;
    }
    tensor.SetName(spec.name);
    weights[spec.name] = std::move(tensor);
  }
  return absl::OkStatus();
}

void MakeSyntheticVideoWeights(const Sam2VideoConfig& config, unsigned seed,
                               WeightMap& weights) {
  unsigned state = seed;
  for (const WeightSpec& spec : GetVideoWeightSpecs(config)) {
    size_t count = 1;
    for (int d : spec.shape) count *= static_cast<size_t>(d);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
      state = state * 1664525u + 1013904223u;
      data[i] = spec.init_scale * (static_cast<float>(state >> 8) /
                                       static_cast<float>(1u << 24) * 2.0f -
                                   1.0f);
    }
    weights[spec.name] =
        TfTensor({.name = spec.name,
                  .type = Type::kFP32,
                  .shape = spec.shape,
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(data)});
  }
}

}  // namespace litert::tensor::examples::sam2_video
