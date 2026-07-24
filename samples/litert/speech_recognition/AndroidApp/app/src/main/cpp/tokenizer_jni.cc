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

#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "include/tokenizers_c.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_examples_asr_HuggingfaceTokenizer_00024Companion_nativeInit(
    JNIEnv* env, jobject thiz, jstring json_payload) {
  const char* json_ptr = env->GetStringUTFChars(json_payload, nullptr);
  std::string json_str(json_ptr);
  env->ReleaseStringUTFChars(json_payload, json_ptr);

  auto tokenizer = tokenizers_new_from_str(json_str.c_str(), json_str.size());
  return reinterpret_cast<jlong>(tokenizer);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_ai_edge_examples_asr_HuggingfaceTokenizer_00024Companion_nativeFree(
    JNIEnv* env, jobject thiz, jlong handle) {
  auto* tokenizer = reinterpret_cast<TokenizerHandle>(handle);
  tokenizers_free(tokenizer);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_ai_edge_examples_asr_HuggingfaceTokenizer_00024Companion_nativeDecode(
    JNIEnv* env, jobject thiz, jlong handle, jintArray ids,
    jboolean skip_special_tokens) {
  auto* tokenizer = reinterpret_cast<TokenizerHandle>(handle);
  if (!tokenizer) return env->NewStringUTF("");

  jsize ids_len = env->GetArrayLength(ids);
  jint* ids_ptr = env->GetIntArrayElements(ids, nullptr);
  std::vector<uint32_t> token_ids(ids_ptr, ids_ptr + ids_len);
  env->ReleaseIntArrayElements(ids, ids_ptr, JNI_ABORT);

  tokenizers_decode(tokenizer, token_ids.data(), token_ids.size(),
                    static_cast<int>(skip_special_tokens));

  const char* decode_str;
  size_t decode_str_len;
  tokenizers_get_decode_str(tokenizer, &decode_str, &decode_str_len);
  std::string decode_str_cpp(decode_str, decode_str_len);
  return env->NewStringUTF(decode_str_cpp.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_ai_edge_examples_asr_HuggingfaceTokenizer_00024Companion_nativeGetVocabSize(
    JNIEnv* env, jobject thiz, jlong handle) {
  auto* tokenizer = reinterpret_cast<TokenizerHandle>(handle);
  if (!tokenizer) return 0;

  size_t vocab_size;
  tokenizers_get_vocab_size(tokenizer, &vocab_size);
  return vocab_size;
}
