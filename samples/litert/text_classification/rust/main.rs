// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `GLiNER2` PII entity extraction / text classification example using `LiteRT` Rust wrappers.
//!
//! This example demonstrates how to use [`litert::LiteRTModelExecutor`] and
//! [`litert::SignatureRunner`] with a `HuggingFace` `Tokenizer` to run a `GLiNER2` PII detection
//! model on input text and decode extracted entity spans.
//!
//! See `tflite_conversion/README.md` for instructions on converting the
//! `fastino/gliner2-privacy-filter-PII-multi` `HuggingFace` model to `.tflite` and exporting
//! `tokenizer.json`.
//!
//! Example usage:
//! ```shell
//! cargo run --release -- \
//!   --model-path=/path/to/gliner2_pii_int8.tflite \
//!   --tokenizer-path=/path/to/tokenizer.json \
//!   --classify-text="Contact Jane Doe at jane.doe@example.com or 555-867-5309."
//! ```

use clap::Parser;
use litert::{
    Error, LiteRTModelExecutor, LiteRtHwAccelerator, Options, SignatureId, SignatureIndex,
    SignatureRunner,
};
use std::ops::Range;
use std::time::Instant;
use tokenizers::tokenizer::Tokenizer;

/// Fixed input subword sequence length expected by the compiled `GLiNER2` `TFLite` model.
const SEQ_LEN: usize = 512;
/// Maximum number of whitespace/punctuation-split text words encoded into the model input.
const MAX_WORDS: usize = 256;
/// Maximum number of entity labels supported in the `GLiNER2` prompt schema.
const MAX_LABELS: usize = 42;
/// Maximum entity span width in words (`end_word - start_word + 1`).
const MAX_WIDTH: usize = 8;
/// Number of logits in the `count_logits` output tensor (`0..MAX_COUNT`).
const MAX_COUNT: usize = 20;
/// Minimum span confidence score required to emit a [`DetectedEntity`].
const DEFAULT_THRESHOLD: f32 = 0.5;

/// `GLiNER2` schema prompt marker token placed before the task name (`"entities"`).
const P_TOKEN: &str = "[P]";
/// `GLiNER2` schema entity marker token placed before each candidate label name.
const E_TOKEN: &str = "[E]";
/// `GLiNER2` separator token placed between the schema prompt and the input text words.
const SEP_TEXT: &str = "[SEP_TEXT]";

/// Default set of 42 Personally Identifiable Information (PII) entity labels queried by the model.
const PII_LABELS: [&str; MAX_LABELS] = [
    "person",
    "full_name",
    "first_name",
    "middle_name",
    "last_name",
    "date_of_birth",
    "email",
    "phone_number",
    "address",
    "street_address",
    "city",
    "state_or_region",
    "postal_code",
    "country",
    "government_id",
    "national_id_number",
    "passport_number",
    "drivers_license_number",
    "license_number",
    "tax_id",
    "tax_number",
    "bank_account",
    "account_number",
    "routing_number",
    "iban",
    "payment_card",
    "card_number",
    "card_expiry",
    "card_cvv",
    "username",
    "ip_address",
    "account_id",
    "sensitive_account_id",
    "password",
    "secret",
    "api_key",
    "access_token",
    "recovery_code",
    "sensitive_date",
    "document_date",
    "expiration_date",
    "transaction_date",
];

/// Command-line arguments for the `GLiNER2` PII text classification example.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the `GLiNER2` `.tflite` model file.
    #[arg(short, long)]
    model_path: String,
    /// Path to the `HuggingFace` `tokenizer.json` file.
    #[arg(short, long)]
    tokenizer_path: String,
    /// Input text string to scan for PII entities.
    #[arg(short, long)]
    classify_text: String,
}

/// Preprocessed and zero-padded input tensors and metadata for a single `GLiNER2` inference call.
struct GraphInputs {
    /// Subword token IDs of length [`SEQ_LEN`], padded with zeros.
    input_ids: Vec<i32>,
    /// Attention mask of length [`SEQ_LEN`] (`1` for valid subwords, `0` for padding).
    attention_mask: Vec<i32>,
    /// Subword start index in `input_ids` for each word (`[MAX_WORDS]`, zero-padded).
    word_positions: Vec<i32>,
    /// Subword index in `input_ids` of each `[E]` label marker (`[MAX_LABELS]`, zero-padded).
    label_positions: Vec<i32>,
    /// Subword index in `input_ids` of the `[P]` prompt marker (`[1]`).
    prompt_position: [i32; 1],
    /// Ordered list of entity label names corresponding to `label_positions`.
    labels: Vec<String>,
    /// Half-open UTF-8 byte offset ranges `start..end` in the original text for each word.
    word_spans: Vec<Range<usize>>,
}

/// A PII entity span extracted and decoded from the model outputs.
#[derive(Debug, Clone, PartialEq)]
struct DetectedEntity {
    /// Predicted PII category label (e.g., `"email"`, `"phone_number"`).
    label: String,
    /// Trimmed substring of the input text covered by the entity span.
    text: String,
    /// Starting UTF-8 byte offset (inclusive) of the entity in the input text.
    start: usize,
    /// Ending UTF-8 byte offset (exclusive) of the entity in the input text.
    end: usize,
    /// Model confidence score in `[0.0, 1.0]` for this span and label.
    confidence: f32,
}

/// Candidate entity span before per-label non-maximum suppression.
#[derive(Debug, Clone, Copy)]
struct CandidateSpan {
    /// Model confidence score for this candidate span.
    confidence: f32,
    /// Starting UTF-8 byte offset (inclusive) in the input text.
    start: usize,
    /// Ending UTF-8 byte offset (exclusive) in the input text.
    end: usize,
}

/// Returns `true` if `c` is a Unicode alphanumeric character or an ASCII underscore (`'_'`).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Returns `true` if `b` is a valid ASCII byte in an email local part.
fn is_email_local_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"._%+-".contains(&b)
}

/// Returns `true` if `b` is a valid ASCII byte in an email domain name.
fn is_email_domain_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'.' || b == b'-'
}

/// Returns `true` if `s` begins with a recognized URL prefix (case-insensitive).
fn starts_with_url_prefix(s: &str) -> bool {
    const URL_PREFIXES: [&str; 3] = ["http://", "https://", "www."];
    URL_PREFIXES
        .iter()
        .any(|prefix| s.get(..prefix.len()).is_some_and(|head| head.eq_ignore_ascii_case(prefix)))
}

/// Result of attempting to match a structured token (such as a URL or email) at the start of a
/// string slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Matched {
    /// A valid token of the given byte length matched at the start of the slice.
    Matched(usize),
    /// No valid token matched; no position within the first `usize` bytes can start one either.
    RejectedPrefix(usize),
}

/// Attempts to match a URL at the start of `s` (typically a whitespace-free chunk).
///
/// Returns [`Matched::Matched`] with the byte length up to the next whitespace (or `s.len()`), or
/// [`Matched::RejectedPrefix`] with the leading byte prefix before the next possible URL start.
fn match_url(s: &str) -> Matched {
    if starts_with_url_prefix(s) {
        return Matched::Matched(s.find(char::is_whitespace).unwrap_or(s.len()));
    }
    let bytes = s.as_bytes();
    for i in 1..bytes.len() {
        if matches!(bytes[i], b'h' | b'H' | b'w' | b'W') && starts_with_url_prefix(&s[i..]) {
            return Matched::RejectedPrefix(i);
        }
    }
    Matched::RejectedPrefix(bytes.len())
}

/// Attempts to match an email address at the start of `s` (typically a whitespace-free chunk).
///
/// Returns [`Matched::Matched`] with the matched byte length, or [`Matched::RejectedPrefix`] with
/// the leading byte prefix ruled out from starting a valid email.
fn match_email(s: &str) -> Matched {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && is_email_local_byte(bytes[i]) {
        i += 1;
    }
    if i == 0 || i >= bytes.len() || bytes[i] != b'@' {
        // Also skip any immediately following bytes that cannot start an email local part
        // (punctuation, non-ASCII, etc.), keeping repeated scans O(N).
        let mut next = i;
        while next < bytes.len() && !is_email_local_byte(bytes[next]) {
            next += 1;
        }
        return Matched::RejectedPrefix(next);
    }

    let after_at = i + 1;
    let domain_start = after_at;
    let mut domain_end = domain_start;
    while domain_end < bytes.len() && is_email_domain_byte(bytes[domain_end]) {
        domain_end += 1;
    }
    for dot_pos in (domain_start + 1..domain_end).rev() {
        if bytes[dot_pos] == b'.' {
            let tld_len = bytes[dot_pos + 1..domain_end]
                .iter()
                .take_while(|b| b.is_ascii_alphabetic())
                .count();
            if tld_len >= 2 {
                return Matched::Matched(dot_pos + 1 + tld_len);
            }
        }
    }
    // Only reject through `@` (not into the domain) so subsequent emails like `a@b+c@example.com`
    // or `a.b@c.d@example.com` still match when scanning resumes after `@`.
    Matched::RejectedPrefix(after_at)
}

/// Matches a `@handle` token (`@` followed by zero or more word characters) at the start of `s`.
fn match_handle(s: &str) -> Option<usize> {
    let rest = s.strip_prefix('@')?;
    let handle_len: usize =
        rest.chars().take_while(|&c| is_word_char(c)).map(char::len_utf8).sum();
    Some(1 + handle_len)
}

/// Matches a contiguous word at the start of `s`, including internal `-` or `_` connectors
/// followed by additional word characters, and returns the matched byte length.
fn match_word_with_connectors(s: &str) -> usize {
    let mut len: usize = s.chars().take_while(|&c| is_word_char(c)).map(char::len_utf8).sum();
    loop {
        let rem = &s[len..];
        let Some(sep) = rem.chars().next() else {
            break;
        };
        if sep != '-' && sep != '_' {
            break;
        }
        let after_sep = &rem[sep.len_utf8()..];
        let seg_len: usize =
            after_sep.chars().take_while(|&c| is_word_char(c)).map(char::len_utf8).sum();
        if seg_len == 0 {
            break;
        }
        len += sep.len_utf8() + seg_len;
    }
    len
}

/// Runs `matcher` on `&chunk[pos..]` unless `pos` lies inside an already-rejected prefix
/// (`pos < *rejected_until`), updating `*rejected_until` on rejection.
fn try_match_with_skip(
    chunk: &str,
    pos: usize,
    rejected_until: &mut usize,
    matcher: fn(&str) -> Matched,
) -> Option<usize> {
    if pos < *rejected_until {
        return None;
    }
    match matcher(&chunk[pos..]) {
        Matched::Matched(len) => Some(len),
        Matched::RejectedPrefix(skip) => {
            *rejected_until = pos + skip;
            None
        }
    }
}

/// Tokenizes a single whitespace-free `chunk` starting at `base_offset` in the original text.
fn tokenize_chunk(chunk: &str, base_offset: usize, result: &mut Vec<(String, Range<usize>)>) {
    let mut pos = 0;
    let mut rejected_url_until = 0;
    let mut rejected_email_until = if chunk.contains('@') { 0 } else { chunk.len() };

    while let Some(first_char) = chunk[pos..].chars().next() {
        let rest = &chunk[pos..];
        let token_len = try_match_with_skip(chunk, pos, &mut rejected_url_until, match_url)
            .or_else(|| try_match_with_skip(chunk, pos, &mut rejected_email_until, match_email))
            .or_else(|| match_handle(rest))
            .unwrap_or_else(|| {
                if is_word_char(first_char) {
                    match_word_with_connectors(rest)
                } else {
                    first_char.len_utf8()
                }
            });

        let end = pos + token_len;
        result.push((chunk[pos..end].to_lowercase(), (base_offset + pos)..(base_offset + end)));
        pos = end;
    }
}

/// Splits `text` into `(lowercased_word, byte_range)` pairs matching `GLiNER2`'s
/// `WhitespaceTokenSplitter` pattern (URLs, emails, `@handles`, hyphen/underscore words, or
/// single non-whitespace characters).
fn split_words(text: &str) -> Vec<(String, Range<usize>)> {
    let mut result = Vec::new();
    let mut pos = 0;
    while let Some(rel_start) = text[pos..].find(|c: char| !c.is_whitespace()) {
        let chunk_start = pos + rel_start;
        let rest = &text[chunk_start..];
        let chunk_len = rest.find(char::is_whitespace).unwrap_or(rest.len());
        tokenize_chunk(&rest[..chunk_len], chunk_start, &mut result);
        pos = chunk_start + chunk_len;
    }
    result
}

/// Encodes `item` with `tokenizer` without adding special tokens and returns its token IDs as
/// `i32`s.
///
/// # Panics
///
/// Panics if the tokenizer fails to encode `item` or if any token ID overflows `i32`.
fn encode_pieces(tokenizer: &Tokenizer, item: &str) -> Vec<i32> {
    tokenizer
        .encode(item, false)
        .expect("Failed to encode piece")
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id).expect("token ID should fit in i32"))
        .collect()
}

/// Constructs the `GLiNER2` schema prompt and word token sequence for `text` and `labels`,
/// returning zero-padded [`GraphInputs`] ready for [`SignatureRunner`] execution.
///
/// # Panics
///
/// Panics if `labels` is empty, exceeds [`MAX_LABELS`], if the encoded schema prompt alone
/// reaches [`SEQ_LEN`], or if any sequence index overflows `i32`.
fn build_inputs(tokenizer: &Tokenizer, text: &str, labels: &[&str]) -> GraphInputs {
    assert!(!labels.is_empty(), "at least one label is required");
    assert!(labels.len() <= MAX_LABELS, "labels exceed MAX_LABELS={MAX_LABELS}");

    let mut ids: Vec<i32> = Vec::new();
    let mut prompt_position: i32 = 0;
    let mut label_positions: Vec<i32> = Vec::with_capacity(labels.len());

    let mut schema: Vec<&str> = vec!["(", P_TOKEN, "entities", "("];
    for &label in labels {
        schema.push(E_TOKEN);
        schema.push(label);
    }
    schema.extend_from_slice(&[")", ")", SEP_TEXT]);

    for (i, item) in schema.iter().enumerate() {
        let pos = i32::try_from(ids.len()).expect("sequence length should fit in i32");
        if i == 1 {
            prompt_position = pos;
        } else if *item == E_TOKEN && i >= 4 {
            label_positions.push(pos);
        }
        ids.extend(encode_pieces(tokenizer, item));
    }

    assert!(
        ids.len() < SEQ_LEN,
        "label prompt alone needs {} subwords; SEQ_LEN={SEQ_LEN}",
        ids.len()
    );

    let mut word_positions: Vec<i32> = Vec::new();
    let mut word_spans: Vec<Range<usize>> = Vec::new();
    for (word, span) in split_words(text) {
        let word_ids = encode_pieces(tokenizer, &word);
        if word_ids.is_empty() {
            continue;
        }
        if word_positions.len() >= MAX_WORDS || ids.len() + word_ids.len() > SEQ_LEN {
            break;
        }
        word_positions.push(i32::try_from(ids.len()).expect("sequence length should fit in i32"));
        word_spans.push(span);
        ids.extend(word_ids);
    }

    let num_real_tokens = ids.len();
    let mut input_ids = vec![0i32; SEQ_LEN];
    input_ids[..num_real_tokens].copy_from_slice(&ids);

    let mut attention_mask = vec![0i32; SEQ_LEN];
    attention_mask[..num_real_tokens].fill(1);

    let mut padded_word_positions = vec![0i32; MAX_WORDS];
    padded_word_positions[..word_positions.len()].copy_from_slice(&word_positions);

    let mut padded_label_positions = vec![0i32; MAX_LABELS];
    padded_label_positions[..label_positions.len()].copy_from_slice(&label_positions);

    GraphInputs {
        input_ids,
        attention_mask,
        word_positions: padded_word_positions,
        label_positions: padded_label_positions,
        prompt_position: [prompt_position],
        labels: labels.iter().map(|&s| s.to_string()).collect(),
        word_spans,
    }
}

/// Decodes [`DetectedEntity`] spans from `count_logits` and `span_scores` above `threshold`,
/// applying greedy non-maximum suppression per label over overlapping byte spans.
///
/// `count_logits` (`argmax > 0`) acts as a binary entity-presence gate matching the exported
/// single-step count embedding in `convert.py`, while individual spans are filtered by `threshold`
/// and per-label greedy non-maximum suppression.
///
/// # Panics
///
/// Panics if `count_logits.len() != MAX_COUNT`, `span_scores.len() != MAX_LABELS * MAX_WORDS *
/// MAX_WIDTH`, `inputs.labels.len() > MAX_LABELS`, or `inputs.word_spans.len() > MAX_WORDS`.
fn decode_entities(
    text: &str,
    inputs: &GraphInputs,
    count_logits: &[f32],
    span_scores: &[f32],
    threshold: f32,
) -> Vec<DetectedEntity> {
    assert_eq!(count_logits.len(), MAX_COUNT);
    assert_eq!(span_scores.len(), MAX_LABELS * MAX_WORDS * MAX_WIDTH);
    assert!(inputs.labels.len() <= MAX_LABELS);
    assert!(inputs.word_spans.len() <= MAX_WORDS);

    let predicted_count = count_logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, val)| !val.is_nan())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    if predicted_count == 0 {
        return Vec::new();
    }

    let num_words = inputs.word_spans.len();
    let mut entities = Vec::new();

    for (k, label) in inputs.labels.iter().enumerate() {
        let mut candidates: Vec<CandidateSpan> = Vec::new();
        for start in 0..num_words {
            for width in 0..MAX_WIDTH {
                let end = start + width;
                if end >= num_words {
                    continue;
                }
                let score = span_scores[(k * MAX_WORDS + start) * MAX_WIDTH + width];
                if score >= threshold {
                    let byte_start = inputs.word_spans[start].start;
                    let byte_end = inputs.word_spans[end].end;
                    if !text[byte_start..byte_end].trim().is_empty() {
                        candidates.push(CandidateSpan {
                            confidence: score,
                            start: byte_start,
                            end: byte_end,
                        });
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

        let mut kept: Vec<CandidateSpan> = Vec::new();
        for cand in candidates {
            if kept.iter().any(|k_span| cand.start < k_span.end && k_span.start < cand.end) {
                continue;
            }
            kept.push(cand);
        }

        for span in kept {
            entities.push(DetectedEntity {
                label: label.clone(),
                text: text[span.start..span.end].trim().to_string(),
                start: span.start,
                end: span.end,
                confidence: span.confidence,
            });
        }
    }

    entities
}

/// High-level wrapper holding the `LiteRT` model executor, signature runner, and tokenizer
/// for `GLiNER2` PII entity extraction.
struct TextClassificationModel {
    /// Owns the `LiteRT` environment, model, and compiled model.
    executor: LiteRTModelExecutor<'static>,
    /// Pre-allocated signature runner for the default model signature.
    runner: SignatureRunner<'static, 'static>,
    /// `HuggingFace` subword tokenizer used to encode schema tokens and text words.
    tokenizer: Tokenizer,
}

impl TextClassificationModel {
    /// Loads the `LiteRT` model from `model_path` on CPU and initializes the tokenizer from
    /// `tokenizer_path`.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if creating the CPU options, compiling the model, or looking up the
    /// default signature fails.
    ///
    /// # Panics
    ///
    /// Panics if `tokenizer_path` cannot be read or parsed as a valid tokenizer JSON file.
    fn try_new(model_path: &str, tokenizer_path: &str) -> Result<Self, Error> {
        let options = Options::create_with_accelerator(LiteRtHwAccelerator::Cpu)?;
        let executor = LiteRTModelExecutor::from_file(model_path, &options)?;
        let runner = executor.signature_runner(SignatureId::default())?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer from file");
        Ok(Self { executor, runner, tokenizer })
    }

    /// Prints the input tensor names of the model's default signature to stdout.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if querying the default signature or its input tensor names fails.
    fn print_input_names(&self) -> Result<(), Error> {
        let signature = self.executor.model().signature(SignatureIndex::DEFAULT)?;
        let input_names = signature.input_names()?;
        for name in input_names {
            println!("Input name: {}", name?);
        }
        Ok(())
    }

    /// Tokenizes `input_text`, executes the `GLiNER2` model, prints the inference duration to
    /// stdout, and returns all detected PII entities.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if binding input tensors, executing the signature runner, or copying
    /// output tensors fails.
    fn run(&mut self, input_text: &str) -> Result<Vec<DetectedEntity>, Error> {
        let inputs = build_inputs(&self.tokenizer, input_text, &PII_LABELS);

        self.runner
            .set_input("input_ids", &inputs.input_ids)?
            .set_input("attention_mask", &inputs.attention_mask)?
            .set_input("word_positions", &inputs.word_positions)?
            .set_input("label_positions", &inputs.label_positions)?
            .set_input("prompt_position", &inputs.prompt_position)?;

        let inference_start = Instant::now();
        self.runner.run()?;
        let inference_duration = inference_start.elapsed();
        println!(
            "Inference completed in {:.2?} ({:.2} ms)",
            inference_duration,
            inference_duration.as_secs_f64() * 1000.0
        );

        let mut count_logits = vec![0.0f32; MAX_COUNT];
        let mut span_scores = vec![0.0f32; MAX_LABELS * MAX_WORDS * MAX_WIDTH];
        self.runner
            .get_output("count_logits", &mut count_logits)?
            .get_output("span_scores", &mut span_scores)?;

        let entities = decode_entities(
            input_text,
            &inputs,
            &count_logits,
            &span_scores,
            DEFAULT_THRESHOLD,
        );
        Ok(entities)
    }
}

/// Entry point that parses CLI arguments, runs `GLiNER2` PII extraction, and prints results.
fn main() {
    let args = Args::parse();
    let mut model = TextClassificationModel::try_new(&args.model_path, &args.tokenizer_path)
        .expect("Failed to initialize the model");
    println!("Model initialized successfully!");
    model.print_input_names().expect("Failed to print input names");
    let entities = model.run(&args.classify_text).expect("Failed to run the model");
    println!("Detected {} entities:", entities.len());
    for entity in &entities {
        println!(
            "  {:<22} {:?} [{}..{}] (confidence: {:.3})",
            entity.label, entity.text, entity.start, entity.end, entity.confidence
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    fn empty_graph_inputs(labels: &[&str], word_spans: Vec<Range<usize>>) -> GraphInputs {
        GraphInputs {
            input_ids: vec![0; SEQ_LEN],
            attention_mask: vec![0; SEQ_LEN],
            word_positions: vec![0; MAX_WORDS],
            label_positions: vec![0; MAX_LABELS],
            prompt_position: [0],
            labels: labels.iter().map(|&s| s.to_string()).collect(),
            word_spans,
        }
    }

    #[gtest]
    fn match_url_handles_case_insensitive_prefixes_and_skips() {
        let url = "https://example.com/a,";
        expect_eq!(match_url(url), Matched::Matched(url.len()));
        expect_eq!(
            match_url("HTTP://Example.com/a rest"),
            Matched::Matched("HTTP://Example.com/a".len())
        );
        expect_eq!(match_url("WWW.example.com"), Matched::Matched("WWW.example.com".len()));
        expect_eq!(match_url("(https://example.com)"), Matched::RejectedPrefix(1));
        expect_eq!(match_url("plain.text"), Matched::RejectedPrefix("plain.text".len()));
        expect_eq!(match_url(""), Matched::RejectedPrefix(0));
    }

    #[gtest]
    fn match_email_handles_valid_backtracking_and_invalid_inputs() {
        let email = "jane.doe+tag@sub.example.co.uk";
        expect_eq!(match_email(email), Matched::Matched(email.len()));
        expect_eq!(
            match_email("jane.doe@example.com."),
            Matched::Matched("jane.doe@example.com".len())
        );
        expect_eq!(
            match_email("jane.doe@example.com-based"),
            Matched::Matched("jane.doe@example.com".len())
        );
        expect_eq!(
            match_email("jane.doe@example.com.1"),
            Matched::Matched("jane.doe@example.com".len())
        );
        expect_eq!(
            match_email("jane.doe@example.com.a"),
            Matched::Matched("jane.doe@example.com".len())
        );
        expect_eq!(match_email("@example.com"), Matched::RejectedPrefix(1));
        expect_eq!(match_email("user@.com"), Matched::RejectedPrefix("user@".len()));
        expect_eq!(match_email("user@domain.c"), Matched::RejectedPrefix("user@".len()));
        expect_eq!(match_email("user@domain"), Matched::RejectedPrefix("user@".len()));
        expect_eq!(match_email("a.b.c!"), Matched::RejectedPrefix("a.b.c!".len()));
        expect_eq!(match_email("a.b.c!d@example.com"), Matched::RejectedPrefix("a.b.c!".len()));
        expect_eq!(match_email(""), Matched::RejectedPrefix(0));
    }

    #[gtest]
    fn split_words_handles_chained_at_signs_wrapped_tokens_and_long_runs() {
        let chained = split_words("a.b@c.d@example.com a@b+c@example.com");
        let chained_words: Vec<&str> = chained.iter().map(|(w, _)| w.as_str()).collect();
        expect_eq!(
            chained_words,
            vec!["a", ".", "b", "@c", ".d@example.com", "a", "@b", "+c@example.com"]
        );

        let chained_sep = split_words("a@!b_c@example.com");
        let chained_sep_words: Vec<&str> = chained_sep.iter().map(|(w, _)| w.as_str()).collect();
        expect_eq!(chained_sep_words, vec!["a", "@", "!", "b_c@example.com"]);

        let wrapped_text = "See (HTTPS://Example.com) and (jane.doe@example.com)!";
        let wrapped = split_words(wrapped_text);
        let wrapped_words: Vec<&str> = wrapped.iter().map(|(w, _)| w.as_str()).collect();
        expect_eq!(
            wrapped_words,
            vec!["see", "(", "https://example.com)", "and", "(", "jane.doe@example.com", ")", "!"]
        );
        for (lower, range) in &wrapped {
            expect_eq!(&wrapped_text[range.clone()].to_lowercase(), lower);
        }

        let long_run = "a.".repeat(20_000);
        expect_eq!(split_words(&long_run).len(), 40_000);

        let long_dotted_with_at = format!("{}@x", "a.".repeat(20_000));
        expect_eq!(split_words(&long_dotted_with_at).len(), 40_001);

        let long_punct_with_at = format!("{}@x", "!a".repeat(20_000));
        expect_eq!(split_words(&long_punct_with_at).len(), 40_001);
    }

    #[gtest]
    fn match_word_with_connectors_handles_hyphens_underscores_and_unicode() {
        expect_eq!(match_word_with_connectors("555-867-5309."), "555-867-5309".len());
        expect_eq!(match_word_with_connectors("first_name_1 "), "first_name_1".len());
        expect_eq!(match_word_with_connectors("word-"), "word".len());
        expect_eq!(match_word_with_connectors("word--next"), "word".len());
        expect_eq!(match_word_with_connectors("Jean-Luc"), "Jean-Luc".len());
        expect_eq!(match_word_with_connectors("Müller_1!"), "Müller_1".len());
    }

    #[gtest]
    fn split_words_tokenizes_urls_emails_handles_and_preserves_utf8_ranges() {
        let text = "Visit HTTPS://Example.com/a, email jane.doe@example.com-based or @Jean_1 @!";
        let tokens = split_words(text);
        let words: Vec<&str> = tokens.iter().map(|(w, _)| w.as_str()).collect();
        expect_eq!(
            words,
            vec![
                "visit",
                "https://example.com/a,",
                "email",
                "jane.doe@example.com",
                "-",
                "based",
                "or",
                "@jean_1",
                "@",
                "!",
            ]
        );
        for (lower, range) in &tokens {
            expect_eq!(&text[range.clone()].to_lowercase(), lower);
        }
    }

    #[gtest]
    fn decode_entities_returns_empty_when_predicted_count_is_zero_or_nan_filtered() {
        let text = "Jane Doe";
        let inputs = empty_graph_inputs(&["person"], vec![0..4, 5..8]);
        let mut count_logits = vec![0.0f32; MAX_COUNT];
        count_logits[0] = 5.0;
        count_logits[1] = f32::NAN;
        let mut span_scores = vec![0.0f32; MAX_LABELS * MAX_WORDS * MAX_WIDTH];
        span_scores[0] = 0.99;

        let entities = decode_entities(text, &inputs, &count_logits, &span_scores, 0.5);
        expect_that!(entities, is_empty());
    }

    #[gtest]
    fn decode_entities_applies_threshold_and_greedy_per_label_nms() {
        let text = "Jane Mary Doe";
        let inputs = empty_graph_inputs(&["person", "first_name"], vec![0..4, 5..9, 10..13]);
        let mut count_logits = vec![0.0f32; MAX_COUNT];
        count_logits[1] = 2.0;

        let mut span_scores = vec![0.0f32; MAX_LABELS * MAX_WORDS * MAX_WIDTH];
        // Label 0 ("person"): full span [0..3] (width=2) has score 0.95; overlapping [0..2]
        // (width=1) has score 0.85 and should be suppressed by NMS; non-overlapping or below
        // threshold spans should be ignored.
        span_scores[(0 * MAX_WORDS + 0) * MAX_WIDTH + 2] = 0.95;
        span_scores[(0 * MAX_WORDS + 0) * MAX_WIDTH + 1] = 0.85;
        span_scores[(0 * MAX_WORDS + 2) * MAX_WIDTH + 0] = 0.40;
        // Out-of-bounds span (start=2, width=2 -> end=4 >= num_words) should be ignored even with
        // high score.
        span_scores[(0 * MAX_WORDS + 2) * MAX_WIDTH + 2] = 0.99;

        // Label 1 ("first_name"): overlapping byte span in a different label is kept.
        span_scores[(1 * MAX_WORDS + 0) * MAX_WIDTH + 0] = 0.90;
        span_scores[(1 * MAX_WORDS + 1) * MAX_WIDTH + 0] = 0.75;

        let entities = decode_entities(text, &inputs, &count_logits, &span_scores, 0.5);
        expect_eq!(
            entities,
            vec![
                DetectedEntity {
                    label: "person".to_string(),
                    text: "Jane Mary Doe".to_string(),
                    start: 0,
                    end: 13,
                    confidence: 0.95,
                },
                DetectedEntity {
                    label: "first_name".to_string(),
                    text: "Jane".to_string(),
                    start: 0,
                    end: 4,
                    confidence: 0.90,
                },
                DetectedEntity {
                    label: "first_name".to_string(),
                    text: "Mary".to_string(),
                    start: 5,
                    end: 9,
                    confidence: 0.75,
                },
            ]
        );
    }

    fn synthetic_tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(
            br###"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
                    {"id": 1, "content": "[P]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                    {"id": 2, "content": "[E]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                    {"id": 3, "content": "[SEP_TEXT]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
                ],
                "normalizer": null,
                "pre_tokenizer": {"type": "WhitespaceSplit"},
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordPiece",
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100,
                    "vocab": {
                        "[PAD]": 0,
                        "[P]": 1,
                        "[E]": 2,
                        "[SEP_TEXT]": 3,
                        "[UNK]": 4,
                        "(": 5,
                        ")": 6,
                        "entities": 7,
                        "person": 8,
                        "phone": 9,
                        "##_number": 10,
                        "jane": 11,
                        "doe": 12,
                        "sub": 13,
                        "##word": 14,
                        "a": 15,
                        "##b": 16,
                        "##c": 17
                    }
                }
            }"###,
        )
        .expect("synthetic tokenizer JSON should parse")
    }

    #[gtest]
    fn build_inputs_lays_out_schema_markers_word_positions_and_zero_padding() {
        let tokenizer = synthetic_tokenizer();
        let text = "Jane subword Doe";
        let inputs = build_inputs(&tokenizer, text, &["phone_number", "person"]);

        // Schema: ( [P] entities ( [E] phone ##_number [E] person ) ) [SEP_TEXT]
        // Text:   jane sub ##word doe
        let expected_ids = vec![5, 1, 7, 5, 2, 9, 10, 2, 8, 6, 6, 3, 11, 13, 14, 12];
        let real_len = expected_ids.len();

        expect_eq!(inputs.input_ids.len(), SEQ_LEN);
        expect_eq!(inputs.input_ids[..real_len], expected_ids);
        expect_that!(inputs.input_ids[real_len..], each(eq(&0)));

        expect_eq!(inputs.attention_mask.len(), SEQ_LEN);
        expect_that!(inputs.attention_mask[..real_len], each(eq(&1)));
        expect_that!(inputs.attention_mask[real_len..], each(eq(&0)));

        expect_eq!(inputs.prompt_position, [1]);

        expect_eq!(inputs.label_positions.len(), MAX_LABELS);
        expect_eq!(inputs.label_positions[..2], [4, 7]);
        expect_that!(inputs.label_positions[2..], each(eq(&0)));

        expect_eq!(inputs.word_positions.len(), MAX_WORDS);
        expect_eq!(inputs.word_positions[..3], [12, 13, 15]);
        expect_that!(inputs.word_positions[3..], each(eq(&0)));

        expect_eq!(inputs.labels, vec!["phone_number", "person"]);
        expect_eq!(inputs.word_spans, vec![0..4, 5..12, 13..16]);
    }

    #[gtest]
    fn build_inputs_truncates_at_max_words() {
        let tokenizer = synthetic_tokenizer();
        let text = "jane ".repeat(MAX_WORDS + 20);
        let inputs = build_inputs(&tokenizer, &text, &["person"]);

        // Schema for one 1-subword label: ( [P] entities ( [E] person ) ) [SEP_TEXT] => 9 tokens.
        let schema_len = 9;
        let expected_real_len = schema_len + MAX_WORDS;

        expect_eq!(inputs.word_spans.len(), MAX_WORDS);
        expect_eq!(inputs.word_positions[0], i32::try_from(schema_len).unwrap());
        expect_eq!(
            inputs.word_positions[MAX_WORDS - 1],
            i32::try_from(schema_len + MAX_WORDS - 1).unwrap()
        );
        expect_that!(inputs.attention_mask[..expected_real_len], each(eq(&1)));
        expect_that!(inputs.attention_mask[expected_real_len..], each(eq(&0)));
        expect_that!(inputs.input_ids[expected_real_len..], each(eq(&0)));
    }

    #[gtest]
    fn build_inputs_truncates_at_seq_len_without_splitting_multi_subword_word() {
        let tokenizer = synthetic_tokenizer();
        // Each "abc" encodes to 3 subwords ([15, 16, 17]). With 9 schema tokens, 503 subwords
        // remain, fitting 167 whole words (501 subwords -> 510 total) before hitting SEQ_LEN=512.
        let text = "abc ".repeat(MAX_WORDS);
        let inputs = build_inputs(&tokenizer, &text, &["person"]);

        let schema_len = 9;
        let expected_words = (SEQ_LEN - schema_len) / 3;
        let expected_real_len = schema_len + expected_words * 3;

        expect_eq!(inputs.word_spans.len(), expected_words);
        expect_eq!(
            inputs.word_positions[expected_words - 1],
            i32::try_from(schema_len + (expected_words - 1) * 3).unwrap()
        );
        expect_that!(inputs.word_positions[expected_words..], each(eq(&0)));
        expect_that!(inputs.attention_mask[..expected_real_len], each(eq(&1)));
        expect_that!(inputs.attention_mask[expected_real_len..], each(eq(&0)));
        expect_that!(inputs.input_ids[expected_real_len..], each(eq(&0)));
    }
}
