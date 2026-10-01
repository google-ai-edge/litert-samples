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

import {resolve} from 'node:path';
import {defineConfig} from 'vite';

export default defineConfig({
  server: {
    port: 5175,
    // e2e tests read clips and weights from $ARTIFACTS (default ../../artifacts) via /@fs/.
    fs: {allow: [resolve(__dirname, '..'), process.env.ARTIFACTS ?? resolve(__dirname, '../../artifacts')]},
  },
  optimizeDeps: {exclude: ['@litertjs/core']},
  build: {target: 'es2022'},
  test: {include: ['test/**/*.test.ts']},
});
