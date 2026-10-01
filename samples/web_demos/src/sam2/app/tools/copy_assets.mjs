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
// ==============================================================================

// Copies the LiteRT.js Wasm runtime into public/ so Vite serves it (runs on
// `npm install`). The SAM 2 model files come from ../tools/build_models.sh.
import {cpSync, mkdirSync} from 'node:fs';
import {resolve} from 'node:path';

const app = resolve(import.meta.dirname, '..');
mkdirSync(resolve(app, 'public/litert-wasm'), {recursive: true});
cpSync(resolve(app, 'node_modules/@litertjs/core/wasm'), resolve(app, 'public/litert-wasm'), {recursive: true});
console.log('copied LiteRT.js wasm -> public/litert-wasm');
