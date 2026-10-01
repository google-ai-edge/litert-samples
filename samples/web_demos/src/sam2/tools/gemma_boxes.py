#!/usr/bin/env python3
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Asks Gemma 4 (LiteRT-LM, `litert-lm serve`) for bounding boxes on an image,
the way the web demo does, and scores them against reference boxes.

The request is an OpenAI chat completion with the image as a data URL and
Gemma's native detection prompt; the reply is a JSON list of
{"box_2d": [ymin, xmin, ymax, xmax], "label": ...} with coordinates in 0..1000
(parsed tolerantly). --schema adds JSON-schema constrained decoding.

  litert-lm serve --api openai --cors-origin http://localhost:5175 &
  python tools/gemma_boxes.py --image frame0.jpg --ask "the soccer ball" [--model gemma-4-e2b,gpu]
      [--ref frame0_ref_boxes.json --ref_key ball] [--schema]
"""
import argparse
import base64
import json
import re
import time
import urllib.request

SCHEMA = {
    'type': 'object',
    'properties': {
        'objects': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'label': {'type': 'string'},
                    'box_2d': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
                               'minItems': 4, 'maxItems': 4},
                },
                'required': ['label', 'box_2d'],
            },
        },
    },
    'required': ['objects'],
}

# Gemma's native detection style (the Gemini convention): a JSON list of
# {"box_2d": [ymin, xmin, ymax, xmax], "label": ...}, coordinates 0-1000.
PROMPT = 'Detect {ask}. Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".'
BOX_RE = re.compile(r'"box_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')
LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]*)"')


def parse_boxes(text):
    """All box_2d entries in the reply (tolerates code fences, prose, partial JSON)."""
    out = []
    for obj in re.split(r'\}\s*,?\s*\{', text):
        m = BOX_RE.search(obj)
        if not m:
            continue
        lab = LABEL_RE.search(obj)
        out.append((lab.group(1) if lab else '', [float(v) for v in m.groups()]))
    return out


def ask(server, model, image_path, what, schema=False):
    data = base64.b64encode(open(image_path, 'rb').read()).decode()
    prompt = PROMPT.format(ask=what)
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{data}'}},
            {'type': 'text', 'text': prompt},
        ]}],
        'temperature': 0,
    }
    if schema:
        body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'boxes', 'schema': SCHEMA}}
    req = urllib.request.Request(f'{server}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    text = out['choices'][0]['message']['content']
    boxes = []
    for label, (y0, x0, y1, x1) in parse_boxes(text):   # [ymin, xmin, ymax, xmax] / 1000
        x0, y0, x1, y1 = (min(max(v / 1000, 0), 1) for v in (x0, y0, x1, y1))
        boxes.append({'label': label, 'box': [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]})
    return boxes, time.time() - t0, text


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:9379')
    ap.add_argument('--model', default='gemma-4-e2b,gpu')
    ap.add_argument('--image', required=True)
    ap.add_argument('--ask', required=True)
    ap.add_argument('--schema', action='store_true', help='constrain the reply with a JSON schema')
    ap.add_argument('--ref', default='')
    ap.add_argument('--ref_key', default='')
    a = ap.parse_args()
    boxes, dt, raw = ask(a.server, a.model, a.image, a.ask, a.schema)
    ref = json.load(open(a.ref))[a.ref_key] if a.ref else None
    print(f'{a.model} · "{a.ask}" · {dt:.1f} s · raw {" ".join(raw.split())[:200]}')
    for b in boxes:
        s = f'  {b["label"]:20s} box (x0,y0,x1,y1) = {[round(v, 3) for v in b["box"]]}'
        if ref:
            s += f'  IoU vs reference {iou(b["box"], ref):.2f}'
        print(s)


if __name__ == '__main__':
    main()
