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

// JS side of the LiteRT runtime in litert_js_runtime.cc: compiles and runs
// .tflite signatures with LiteRT.js on its WebGPU device and manages the
// WebGPU buffers that back LiteRT tensor buffers.
//
// The embedding page passes the LiteRT.js module in the module factory
// options: createSam2Chain({litert: {core, precision?}}), after
// core.loadLiteRt(...). The WebGPU device is LiteRT.js's own (created by its
// first WebGPU compile), so tensors are shared with it without copies.

addToLibrary({
  // Module.lrt.buffers maps the ids the C++ side hands out (frameBuffer(),
  // outputBuffer()) to GPUBuffers, for zero-copy frame upload and display.
  $lrt__postset: "Module['lrt'] = lrt;",
  $lrt: {
    models: new Map(),
    buffers: new Map(),
    tensors: new Map(),  // bufId -> Map(shapeKey -> core.Tensor)
    next: 1,
    staging: new Map(),  // size -> MAP_READ buffer
    prof: {mode: null, records: []},  // see lrtjs_run
    ctx() {
      const l = Module['litert'];
      if (!l || !l.core) throw new Error('litert-js: Module.litert = {core} not set');
      const device = l.core.getWebGpuDevice();
      return {core: l.core, device, precision: l.precision ?? 'fp16'};
    },
  },

  lrtjs_compile__deps: ['$lrt'],
  lrtjs_compile__async: true,
  lrtjs_compile: async function(bytesPtr, size, gpu) {
    try {
      const {core, precision} = lrt.ctx();
      const bytes = HEAPU8.slice(bytesPtr, bytesPtr + size);
      const model = await core.loadAndCompile(bytes, gpu
        ? {accelerator: 'webgpu', gpuOptions: {precision}} : {accelerator: 'wasm'});
      const details = {};
      for (const [key, sig] of Object.entries(model.signatures)) {
        details[key] = Object.fromEntries(sig.getInputDetails().map((d) => [
          d.name,
          {shape: Array.from(d.shape), dtype: d.dtype, shapeKey: `${d.dtype}:${d.shape.join('x')}`},
        ]));
      }
      const id = lrt.next++;
      lrt.models.set(id, {model, details});
      return id;
    } catch (e) {
      console.error('litert-js compile:', e);
      return -1;
    }
  },

  lrtjs_destroy_model__deps: ['$lrt'],
  lrtjs_destroy_model: function(id) {
    lrt.models.get(id)?.model.delete();
    lrt.models.delete(id);
  },

  lrtjs_fully_accelerated__deps: ['$lrt'],
  lrtjs_fully_accelerated: function(id) {
    return lrt.models.get(id)?.model.isFullyAccelerated ? 1 : 0;
  },

  lrtjs_run__deps: ['$lrt'],
  lrtjs_run__async: true,
  lrtjs_run: async function(id, sigPtr, inNamesPtr, inIdsPtr, nIn, outNamesPtr, outIdsPtr, outBytesPtr, nOut) {
    const {core, device} = lrt.ctx();
    const entry = lrt.models.get(id);
    const key = UTF8ToString(sigPtr);
    const inNames = UTF8ToString(inNamesPtr).split(',');
    const outNames = UTF8ToString(outNamesPtr).split(',');
    const inputs = {};
    let out = null;
    // Profiling (Module.lrt.prof.mode, set by the page): 'cpu' records how
    // long run() takes to return; 'gpu' also waits for the GPU after each
    // signature (serializes the queue, so only use it to split the time).
    const prof = lrt.prof;
    const p0 = prof.mode ? performance.now() : 0;
    try {
      const details = entry.details[key];
      for (let i = 0; i < nIn; i++) {
        const name = inNames[i];
        const d = details[name];
        const bufId = HEAP32[(inIdsPtr >> 2) + i];
        let byShape = lrt.tensors.get(bufId);
        if (!byShape) {
          byShape = new Map();
          lrt.tensors.set(bufId, byShape);
        }
        let t = byShape.get(d.shapeKey);
        if (!t) {
          t = new core.Tensor(lrt.buffers.get(bufId), d.shape, d.dtype);
          byShape.set(d.shapeKey, t);
        }
        inputs[name] = t;
      }
      out = await entry.model.run(key, inputs);
      const p1 = prof.mode ? performance.now() : 0;
      // Results land in the chain's own buffers (the output tensors LiteRT.js
      // allocates are released right after): one GPU copy per output.
      const enc = device.createCommandEncoder();
      for (let i = 0; i < nOut; i++) {
        const dst = lrt.buffers.get(HEAP32[(outIdsPtr >> 2) + i]);
        const bytes = HEAP32[(outBytesPtr >> 2) + i];
        enc.copyBufferToBuffer(out[outNames[i]].toGpuBuffer(), 0, dst, 0, bytes);
      }
      device.queue.submit([enc.finish()]);
      if (prof.mode) {
        const p2 = performance.now();
        let gpu = NaN;
        if (prof.mode === 'gpu') {
          await device.queue.onSubmittedWorkDone();
          gpu = performance.now() - p0;
        }
        prof.records.push({key, run: p1 - p0, copy: p2 - p1, gpu});
      }
      return 1;
    } catch (e) {
      console.error(`litert-js run ${key}:`, e);
      return 0;
    } finally {
      if (out) for (const t of Object.values(out)) t.delete();
    }
  },

  lrtjs_create_buffer__deps: ['$lrt'],
  lrtjs_create_buffer: function(bytes) {
    const {device} = lrt.ctx();
    const buf = device.createBuffer({
      size: Math.max(4, Math.ceil(bytes / 4) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const id = lrt.next++;
    lrt.buffers.set(id, buf);
    return id;
  },

  lrtjs_destroy_buffer__deps: ['$lrt'],
  lrtjs_destroy_buffer: function(id) {
    const byShape = lrt.tensors.get(id);
    if (byShape) {
      for (const t of byShape.values()) t.delete();
      lrt.tensors.delete(id);
    }
    lrt.buffers.get(id)?.destroy();
    lrt.buffers.delete(id);
  },

  lrtjs_write__deps: ['$lrt'],
  lrtjs_write: function(id, src, bytes) {
    const {device} = lrt.ctx();
    device.queue.writeBuffer(lrt.buffers.get(id), 0, HEAPU8.buffer, src, Math.ceil(bytes / 4) * 4);
  },

  lrtjs_read__deps: ['$lrt'],
  lrtjs_read__async: true,
  lrtjs_read: async function(id, dst, bytes) {
    const {device} = lrt.ctx();
    const size = Math.ceil(bytes / 4) * 4;
    let staging = lrt.staging.get(size);
    if (!staging || staging.mapState !== 'unmapped') {
      staging = device.createBuffer({size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
      lrt.staging.set(size, staging);
    }
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(lrt.buffers.get(id), 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    try {
      await staging.mapAsync(GPUMapMode.READ);
      HEAPU8.set(new Uint8Array(staging.getMappedRange(0, size), 0, bytes), dst);
      staging.unmap();
      return 1;
    } catch (e) {
      console.error('litert-js read:', e);
      return 0;
    }
  },

  lrtjs_read_many__deps: ['$lrt'],
  lrtjs_read_many__async: true,
  lrtjs_read_many: async function(idsPtr, bytesPtr, count, dst) {
    const {device} = lrt.ctx();
    let totalSize = 0;
    const items = [];
    for (let i = 0; i < count; i++) {
      const id = HEAP32[(idsPtr >> 2) + i];
      const bytes = HEAP32[(bytesPtr >> 2) + i];
      const size = Math.ceil(bytes / 4) * 4;
      items.push({id, bytes, size, srcOffset: totalSize});
      totalSize += size;
    }
    let staging = lrt.staging.get(totalSize);
    if (!staging || staging.mapState !== 'unmapped') {
      staging = device.createBuffer({
        size: Math.max(4, totalSize),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      lrt.staging.set(totalSize, staging);
    }
    const enc = device.createCommandEncoder();
    for (const it of items) {
      enc.copyBufferToBuffer(lrt.buffers.get(it.id), 0, staging, it.srcOffset, it.size);
    }
    device.queue.submit([enc.finish()]);
    try {
      await staging.mapAsync(GPUMapMode.READ);
      const range = staging.getMappedRange(0, totalSize);
      let dstOffset = dst;
      for (const it of items) {
        HEAPU8.set(new Uint8Array(range, it.srcOffset, it.bytes), dstOffset);
        dstOffset += it.bytes;
      }
      staging.unmap();
      return 1;
    } catch (e) {
      console.error('litert-js read_many:', e);
      return 0;
    }
  },
});
