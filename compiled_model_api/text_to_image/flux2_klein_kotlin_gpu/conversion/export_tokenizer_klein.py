"""Stage everything the app needs to tokenize and embed a prompt on device.

The twelve graphs cover the transformer, but the prompt has to become
`inputs_embeds` before the first one runs. Two prompt-dependent tensors are all
that `gen_prep_klein.py` bakes in -- `inputs_embeds` and `enc_mask` -- so
shipping the tokenizer and the embedding table is enough to make the prompt
editable in the app. Everything else it stages (both rotary tables, the timestep
embeddings, the scheduler deltas, the initial latents, the tail permutations)
depends only on positions, the schedule or the seed.

Written into `<out>/`:

    qwen_vocab.txt      one byte-level BPE token per line, ordered by id
    qwen_merges.txt     one merge rule per line, in rank order
    qwen_special.txt    `<token>\\t<id>` for every added token
    qwen_embed_fp16.bin [vocab_size, hidden] fp16, row-major (the lookup table)
    tokenizer_fixture.txt  `<prompt>\\t<id>,<id>,...` cases the Kotlin port must
                           reproduce exactly

The embedding table is fp16 and memory-mapped by the app, exactly as the RWKV-7
sample does: a `GATHER` over 151936 rows is not a GPU op, and the row is the
graph's input anyway.

Usage:
    python export_tokenizer_klein.py --out klein_tokenizer
"""

import argparse
import os

import numpy as np
import torch

REPO = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_OUT = "klein_tokenizer"

# The prompts the Kotlin tokenizer is fixture-tested against. Cover ASCII, the
# leading-space rule, punctuation, digits, CJK and emoji, because byte-level BPE
# gets each of them wrong in a different way.
FIXTURE_PROMPTS = [
    "a red apple on a wooden table, studio lighting",
    "turn the apple into a green apple",
    "A",
    " leading space",
    "trailing space ",
    "punctuation!? -- yes; no.",
    "digits 1234567890 and 42",
    "new\nline",
    "CJK 日本語のテキスト",
    "emoji 🍎🚀 mixed with text",
    "'s 't 're contractions",
    "  double  spaces  everywhere  ",
    "don't split THIS wrongly",
    "3.14159 and -42 and 1e-9",
    "tab\tseparated\tvalues",
    "Ⅷ Ⅸ roman numerals and ½ ¾",
    "mixed日本語123English",
    "«guillemets» and — em dash …",
    "a" * 300,
    "\u200bzero width\u200b space",
]


def export_tokenizer(tokenizer, out_dir):
    """Writes the vocabulary, the merge table and the added tokens.

    The merge ranks are read from `tokenizer.json` rather than the tokenizer
    object: the fast tokenizer keeps them inside its Rust model and exposes no
    `bpe_ranks`.

    Args:
        tokenizer: A `Qwen2Tokenizer`.
        out_dir: Directory to write into.
    """
    import json

    from huggingface_hub import hf_hub_download

    spec = json.load(open(hf_hub_download(REPO, "tokenizer/tokenizer.json")))
    vocab = spec["model"]["vocab"]
    merges = spec["model"]["merges"]

    added_max = max(tokenizer.get_added_vocab().values())
    size = max(max(vocab.values()), added_max) + 1
    tokens = [""] * size
    for token, index in vocab.items():
        tokens[index] = token
    for token, index in tokenizer.get_added_vocab().items():
        tokens[index] = token
    for token in tokens:
        assert not any(c.isspace() for c in token), repr(token)

    vocab_path = os.path.join(out_dir, "qwen_vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))

    merges_path = os.path.join(out_dir, "qwen_merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("\n".join(
            merge if isinstance(merge, str) else " ".join(merge)
            for merge in merges))

    added = tokenizer.get_added_vocab()
    special_path = os.path.join(out_dir, "qwen_special.txt")
    with open(special_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{tok}\t{idx}" for tok, idx in sorted(
            added.items(), key=lambda kv: kv[1])))

    print(f"[tokenizer] {size} tokens, {len(merges)} merges, "
          f"{len(added)} added")


def export_embeddings(out_dir):
    """Writes the Qwen3 token-embedding table as fp16, row-major.

    Args:
        out_dir: Directory to write into.

    Returns:
        The (vocab_size, hidden) shape of the table.
    """
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained(
        REPO, subfolder="text_encoder", torch_dtype=torch.float32).eval()
    weight = model.model.embed_tokens.weight.detach()
    path = os.path.join(out_dir, "qwen_embed_fp16.bin")
    weight.to(torch.float16).numpy().astype("<f2").tofile(path)
    megabytes = os.path.getsize(path) / 1e6
    print(f"[embed] {tuple(weight.shape)} fp16 -> {megabytes:.0f} MB")
    return tuple(weight.shape)


def export_fixture(tokenizer, out_dir):
    """Writes prompt/id pairs the Kotlin tokenizer must reproduce exactly.

    Only the prompt body is tokenized: the chat template's wrapper tokens are
    added tokens with fixed ids, so the app can splice them in directly.

    Args:
        tokenizer: A `Qwen2Tokenizer`.
        out_dir: Directory to write into.
    """
    lines = []
    for prompt in FIXTURE_PROMPTS:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        escaped = (prompt.replace("\\", "\\\\")
                   .replace("\n", "\\n").replace("\t", "\\t"))
        lines.append(f"{escaped}\t{','.join(str(i) for i in ids)}")
    path = os.path.join(out_dir, "tokenizer_fixture.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[fixture] {len(lines)} cases -> {path}")


def print_template(tokenizer):
    """Prints the wrapper ids the app splices around the prompt body.

    Args:
        tokenizer: A `Qwen2Tokenizer`.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    body = tokenizer("X", add_special_tokens=False)["input_ids"]
    cut = ids.index(body[0])
    print(f"[template] {rendered!r}")
    print(f"[template] prefix ids {ids[:cut]}")
    print(f"[template] suffix ids {ids[cut + len(body):]}")


def main():
    """Stages the tokenizer, the embedding table and the fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    export_tokenizer(tokenizer, args.out)
    export_fixture(tokenizer, args.out)
    print_template(tokenizer)
    export_embeddings(args.out)
    print(f"[done] staged into {args.out}/")


if __name__ == "__main__":
    main()
