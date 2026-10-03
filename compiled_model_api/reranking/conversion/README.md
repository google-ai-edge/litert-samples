# Qwen3-Reranker-0.6B → LiteRT CompiledModel GPU (fully-GPU, fp16)

Reproduces the `.tflite` reranker graph: a fully GPU-compatible re-authoring of [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (Apache-2.0).

The reranker is the **same Qwen3-0.6B decoder** as [`Qwen3-Embedding-0.6B`](../../semantic_similarity/conversion), so the entire GPU-clean re-authoring is identical (host-embed input, GQA `cat`-repeat, **max-normalized SafeRMS** for the deep-stack fp16 overflow, baked RoPE / causal mask, every tensor ≤4D — see the embedding `conversion/README`). Only the head differs.

## The reranker head

Qwen3-Reranker scores a `(query, document)` pair by a single forward pass over the prompt `PREFIX + "<Instruct>:… <Query>:… <Document>:…" + SUFFIX`, then reads the last-token logits for the tokens `"yes"` (9693) and `"no"` (2152) and takes `softmax → P(yes)`.

Since `tie_word_embeddings=True`, those two logit rows are just two rows of the token-embedding table. So the graph bakes a tiny **2-logit head** = `embed_tokens.weight[[no_id, yes_id]]` (`[2,1024]`) after the final norm, and outputs `[1,L,2]`. The host takes the softmax at the last real token.

`padding_side` note: the official pipeline left-pads + masks; this graph **right-pads and pools the last real token** (position `real_len-1`) — with causal attention the pooled token never sees the trailing pad, so the result is identical and no attention mask is needed.

## Result (device-verified, Pixel 8a / Tensor G3)

- CPU parity vs HF: relevant pair P(yes) **0.9995**, irrelevant **0.0000** (|Δ| = 0.00000)
- Device fp16 P(yes): ref **0.9995** / dev **0.9994** (|Δ| = 0.00006) — `fp16 OK`
- GPU compile ~4 s (cached; identical architecture to the embedder), run fast, `.tflite` GPU-CLEAN

`L=256` (docs need more room than the embedder's 128); fp16 holds because a real prompt is short (~85 tokens) and causal attention means the pooled token sees only those.

## Reproduce

```bash
hf download Qwen/Qwen3-Reranker-0.6B --local-dir ./Qwen3-Reranker-0.6B
python build_qwen3rerank.py --L 256    # CPU parity (P(yes) vs HF) + qwen3rerank_gpu.tflite
python check_qwen3rerank.py            # op-check (GPU-CLEAN) + cast to fp16
python export_embeddings.py ./Qwen3-Reranker-0.6B .   # embeddings_fp16.bin for host lookup
python make_fixture_rerank.py          # device probe fixture
# device probe, then:
python compare_rerank.py probe_out_0.bin
```

The `.tflite` files are not committed; the sample fetches them at build time.
