// Copyright 2026 Daisuke Majima. All Rights Reserved.
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
// =============================================================================

// Qwen3 tokenizer (byte-level BPE) in pure Swift — enough to feed the Bonsai
// text encoder on device. Loads vocab.json + merges.txt (the tokenizer/ folder
// of the HF repo). The chat template is applied structurally: the graph input
// is [<|im_start|>] + BPE("user\n" + prompt) + a fixed assistant suffix, which
// reproduces transformers' apply_chat_template(..., enable_thinking=False)
// exactly — the <|im_end|> added token forces a pre-tokenization split, so the
// suffix ids are prompt-independent constants, while "user\n" must be BPE'd
// TOGETHER with the prompt (a prompt starting with whitespace merges across
// that boundary). Verified against the Python tokenizer on a 26-case golden
// set (ASCII/CJK/emoji/NFD/whitespace edges) before shipping.
//
// Not handled (irrelevant for an image-prompt box): special-token strings
// typed literally inside the prompt are tokenized as plain text.

import Foundation

final class QwenTokenizer {
    static let seqLen = 256
    static let padID: Int32 = 151643        // <|endoftext|>
    static let imStartID: Int32 = 151644    // <|im_start|>
    // "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let suffixIDs: [Int32] = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]

    private let vocab: [String: Int32]
    private let ranks: [String: Int]
    private let regex: NSRegularExpression
    private var cache: [String: [Int32]] = [:]

    // Qwen2 pre-tokenization pattern, verbatim from tokenizer.json.
    private static let pattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" +
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    // GPT-2 bytes-to-unicode: printable bytes map to themselves, the rest to
    // U+0100... so every byte is a distinct printable scalar and " " never
    // appears inside a symbol (which makes "left right" rank keys unambiguous).
    private static let byteChar: [Character] = {
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0...255 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(UnicodeScalar(c)!) }
        return table
    }()

    init(vocabURL: URL, mergesURL: URL) throws {
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL))
        guard let dict = raw as? [String: Int] else {
            throw NSError(domain: "QwenTokenizer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "vocab.json is not {token: id}"])
        }
        var v: [String: Int32] = [:]
        v.reserveCapacity(dict.count)
        for (k, id) in dict { v[k] = Int32(id) }
        vocab = v

        var r: [String: Int] = [:]
        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
        var rank = 0
        merges.enumerateLines { line, _ in
            if line.hasPrefix("#") || line.isEmpty { return }
            r[line] = rank
            rank += 1
        }
        ranks = r
        regex = try NSRegularExpression(pattern: Self.pattern)
    }

    struct Encoded {
        let ids: [Int32]
        let mask: [Int32]
        let promptTokenCount: Int
    }

    /// Full graph input for one user prompt: template + pad to 256, right-pad
    /// mask. Over-long prompts are truncated so the assistant suffix survives.
    func encodePrompt(_ prompt: String) -> Encoded {
        var body = encode("user\n" + prompt)
        let maxBody = Self.seqLen - 1 - Self.suffixIDs.count
        if body.count > maxBody { body = Array(body.prefix(maxBody)) }
        var ids = [Self.imStartID] + body + Self.suffixIDs
        let real = ids.count
        ids.append(contentsOf: [Int32](repeating: Self.padID, count: Self.seqLen - real))
        let mask = [Int32](repeating: 1, count: real)
                 + [Int32](repeating: 0, count: Self.seqLen - real)
        return Encoded(ids: ids, mask: mask, promptTokenCount: body.count)
    }

    /// Byte-level BPE of plain text (no special-token splitting).
    func encode(_ text: String) -> [Int32] {
        let nfc = text.precomposedStringWithCanonicalMapping
        let ns = nfc as NSString
        var out: [Int32] = []
        let matches = regex.matches(in: nfc, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            out.append(contentsOf: bpe(ns.substring(with: m.range)))
        }
        return out
    }

    private func bpe(_ pretoken: String) -> [Int32] {
        if let hit = cache[pretoken] { return hit }
        var word: [String] = pretoken.utf8.map { String(Self.byteChar[Int($0)]) }
        while word.count > 1 {
            var best = Int.max
            var at = -1
            for i in 0..<(word.count - 1) {
                if let r = ranks[word[i] + " " + word[i + 1]], r < best {
                    best = r
                    at = i
                }
            }
            if at < 0 { break }
            let a = word[at], b = word[at + 1]
            var merged: [String] = []
            merged.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == a, word[i + 1] == b {
                    merged.append(a + b)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }
        // Byte-level alphabet: every symbol/merge result exists in the vocab.
        let ids = word.compactMap { vocab[$0] }
        cache[pretoken] = ids
        return ids
    }
}
