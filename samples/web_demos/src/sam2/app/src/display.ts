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

// Shows the pipeline's composite on screen. The composite ([1,H,W,3] fp32,
// produced in-graph by the C++ pipeline) already contains the frame, masks,
// outlines and effect; this is only a blit of that GPU buffer to a WebGPU
// canvas (no readback), plus click markers on a 2D canvas above it.

const BLIT = /* wgsl */ `
struct Size { w: u32, h: u32 };
@group(0) @binding(0) var<storage, read> rgb: array<f32>;
@group(0) @binding(1) var<uniform> size: Size;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = min(u32(pos.x), size.w - 1u);
  let y = min(u32(pos.y), size.h - 1u);
  let i = (y * size.w + x) * 3u;
  return vec4f(clamp(vec3f(rgb[i], rgb[i + 1u], rgb[i + 2u]), vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export interface Marker {
  nx: number;
  ny: number;
  label: 0 | 1 | 2 | 3;  // 2 / 3: a box's top-left / bottom-right corner
  color: [number, number, number];
}

/** A box being dragged (normalized frame coords). */
export interface DraftBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  color: [number, number, number];
}

export class GpuView {
  private readonly ctx: GPUCanvasContext;
  private readonly pipeline: GPURenderPipeline;
  private readonly uniform: GPUBuffer;
  private bind: {buffer: GPUBuffer; group: GPUBindGroup} | null = null;

  constructor(readonly canvas: HTMLCanvasElement, readonly marks: HTMLCanvasElement,
              private readonly device: GPUDevice) {
    this.ctx = canvas.getContext('webgpu')!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({device, format, alphaMode: 'opaque'});
    const module = device.createShaderModule({code: BLIT});
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {module, entryPoint: 'vs'},
      fragment: {module, entryPoint: 'fs', targets: [{format}]},
    });
    this.uniform = device.createBuffer({size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  }

  resize(width: number, height: number) {
    for (const c of [this.canvas, this.marks]) {
      c.width = width;
      c.height = height;
    }
    this.device.queue.writeBuffer(this.uniform, 0, new Uint32Array([width, height]));
  }

  /** Draws the composite buffer ([H, W, 3] fp32). */
  show(rgb: GPUBuffer) {
    if (this.bind?.buffer !== rgb) {
      this.bind = {buffer: rgb, group: this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [{binding: 0, resource: {buffer: rgb}}, {binding: 1, resource: {buffer: this.uniform}}],
      })};
    }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({colorAttachments: [{
      view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store',
      clearValue: {r: 0, g: 0, b: 0, a: 1}}]});
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind.group);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Prompt markers: positive click = object colour with "+", negative = red
   * with "−", a box = its outline in the object colour; `draft` is a box being
   * dragged.
   */
  drawMarkers(markers: Marker[], draft?: DraftBox) {
    const g = this.marks.getContext('2d')!;
    const {width: W, height: H} = this.marks;
    g.clearRect(0, 0, W, H);
    const s = Math.max(W / 960, 0.6);
    const rect = (x0: number, y0: number, x1: number, y1: number, color: [number, number, number], dashed: boolean) => {
      g.save();
      g.lineWidth = 2.5 * s;
      g.setLineDash(dashed ? [8 * s, 6 * s] : []);
      g.strokeStyle = 'rgba(0,0,0,0.55)';
      g.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
      g.lineWidth = 1.5 * s;
      g.strokeStyle = `rgb(${color.join(',')})`;
      g.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
      g.restore();
    };
    for (const m of markers) {
      if (m.label === 2) {
        const br = markers.find((b) => b.label === 3 && b.color === m.color);
        if (br) rect(m.nx, m.ny, br.nx, br.ny, m.color, false);
      }
    }
    if (draft) {
      rect(Math.min(draft.x0, draft.x1), Math.min(draft.y0, draft.y1), Math.max(draft.x0, draft.x1),
          Math.max(draft.y0, draft.y1), draft.color, true);
    }
    for (const m of markers) {
      if (m.label > 1) continue;
      const x = m.nx * W, y = m.ny * H;
      g.beginPath();
      g.arc(x, y, 9 * s, 0, Math.PI * 2);
      g.fillStyle = m.label === 1 ? `rgb(${m.color.join(',')})` : '#e5484d';
      g.fill();
      g.lineWidth = 2.5 * s;
      g.strokeStyle = '#fff';
      g.stroke();
      g.beginPath();
      g.moveTo(x - 4.5 * s, y);
      g.lineTo(x + 4.5 * s, y);
      if (m.label === 1) {
        g.moveTo(x, y - 4.5 * s);
        g.lineTo(x, y + 4.5 * s);
      }
      g.stroke();
    }
  }
}
