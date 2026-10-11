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

package com.google.ai.edge.examples.voice_assistant

/**
 * The one error type of the sample's engines. [code] says what kind of failure it is, as the
 * screen and the log show it:
 * - `INVALID_INPUT`: the call's input cannot be used (audio longer than the window, text with
 *   nothing to say);
 * - `MODEL_BUSY`: a second call while one runs (one call at a time per model);
 * - `MODEL_CLOSED`: the model is closing or closed;
 * - `INITIALIZATION_FAILED`: a model file does not load or does not match what the engine expects;
 * - `INFERENCE_FAILED`: the runtime failed during a call;
 * - `DOWNLOAD_FAILED`: a model file is not on the phone and could not be downloaded;
 * - `SESSION_INVALIDATED`: a chat conversation was cancelled or failed (open a new one);
 * - `SLOW_CONSUMER`: the chat stream's collector fell too far behind (the generation is cancelled);
 * - `NATIVE_STOP_TIMEOUT`: a generation did not stop in time, the chat engine is left unusable.
 */
class VoiceAssistantException(val code: String, message: String, cause: Throwable? = null) :
  Exception(message, cause)
