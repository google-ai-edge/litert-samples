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

"""A small OpenAI-compatible server for the demo's "Ask Gemma", on the
LiteRT-LM Python API (`litert_lm.Engine`), tuned for detection latency.

`litert-lm serve` answers the same requests, but it builds every conversation
with constrained decoding enabled (~0.8 s of setup per request, and slower
decoding), even when no response format is asked for. This server keeps one
warm engine per model (language model and vision encoder on the GPU), creates
plain conversations, and streams tokens so the page can start SAM 2 on each
box as soon as Gemma has written it.

Endpoints (the subset the demo uses):
  GET  /v1/models
  POST /v1/chat/completions   {model, messages: [{role: "user", content: [
                                 {type: "image_url", image_url: {url: "data:image/jpeg;base64,..."}},
                                 {type: "text", text: "..."}]}],
                               temperature?, stream?}
The model id may carry a backend suffix ("gemma-4-e4b,gpu"); it is ignored.

  python tools/gemma_server.py [--port 9379] [--cors-origin http://localhost:5175 ...]
"""
import argparse
import base64
import json
import pathlib
import struct
import sys
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litert_lm

MODELS_DIR = pathlib.Path.home() / '.litert-lm' / 'models'


def _png(w: int, h: int) -> bytes:
  """A plain grey PNG (to warm up the vision encoder)."""
  def chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
  raw = b''.join(b'\0' + b'\x80' * (3 * w) for _ in range(h))
  return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
          chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


class Engines:
  """Warm engines by model id, each used by one request at a time. With
  keep_one (the default), loading a model closes the others: two Gemma 4
  engines do not fit next to a browser in 16 GB, and paging them makes
  decoding several times slower."""

  def __init__(self, ids: list[str], keep_one: bool = True):
    self.keep_one = keep_one
    self.ids = [i for i in ids if (MODELS_DIR / i / 'model.litertlm').exists()]
    self._engines: dict[str, litert_lm.Engine] = {}
    self._locks = {i: threading.Lock() for i in self.ids}
    self._load_lock = threading.Lock()

  def get(self, model_id: str) -> tuple[litert_lm.Engine, threading.Lock]:
    if model_id not in self._locks:
      raise KeyError(model_id)
    with self._load_lock:
      if model_id not in self._engines:
        if self.keep_one:
          for other in list(self._engines):
            with self._locks[other]:  # waits for its request to finish
              self._engines.pop(other).close()
            print(f'unloaded {other}', flush=True)
        t = time.time()
        self._engines[model_id] = litert_lm.Engine(
            str(MODELS_DIR / model_id / 'model.litertlm'),
            backend=litert_lm.Backend.GPU(), vision_backend=litert_lm.Backend.GPU())
        print(f'loaded {model_id} in {time.time() - t:.1f} s', flush=True)
    return self._engines[model_id], self._locks[model_id]

  def warm_up(self, model_id: str):
    """Loads the engine and runs one image request (first runs compile GPU programs)."""
    t = time.time()
    engine, lock = self.get(model_id)
    blob = base64.b64encode(_png(64, 36)).decode()
    with lock, engine.create_conversation(sampler_config=litert_lm.SamplerConfig(temperature=0.0)) as conv:
      conv.send_message({'role': 'user', 'content': [{'type': 'image', 'blob': blob},
                                                     {'type': 'text', 'text': 'Say OK.'}]})
    print(f'warmed up {model_id} in {time.time() - t:.1f} s', flush=True)


def _to_litert_content(content) -> list[dict]:
  """OpenAI message content -> LiteRT-LM content parts."""
  if isinstance(content, str):
    return [{'type': 'text', 'text': content}]
  parts = []
  for p in content:
    if p.get('type') == 'text':
      parts.append({'type': 'text', 'text': p['text']})
    elif p.get('type') == 'image_url':
      url = p['image_url']['url']
      if not url.startswith('data:') or ';base64,' not in url:
        raise ValueError('only base64 data URLs are supported for images')
      parts.append({'type': 'image', 'blob': url.split(',', 1)[1]})
    else:
      raise ValueError(f'unsupported content part {p.get("type")!r}')
  return parts


def _chunk_text(chunk) -> str:
  if isinstance(chunk, dict):
    return ''.join(c.get('text', '') for c in chunk.get('content', []) if isinstance(c, dict))
  return str(chunk)


class Handler(BaseHTTPRequestHandler):
  engines: Engines
  origins: set[str]
  protocol_version = 'HTTP/1.1'

  def log_message(self, fmt, *args):
    pass

  def _cors(self):
    origin = self.headers.get('Origin')
    if origin and ('*' in self.origins or origin in self.origins):
      self.send_header('Access-Control-Allow-Origin', origin)
      self.send_header('Vary', 'Origin')

  def _json(self, code: int, obj):
    body = json.dumps(obj).encode()
    self.send_response(code)
    self._cors()
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_OPTIONS(self):
    self.send_response(204)
    self._cors()
    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    self.send_header('Content-Length', '0')
    self.end_headers()

  def do_GET(self):
    if self.path.rstrip('/') != '/v1/models':
      return self._json(404, {'error': {'message': 'not found'}})
    self._json(200, {'object': 'list', 'data': [
        {'id': i, 'object': 'model', 'owned_by': 'litert-lm'} for i in self.engines.ids]})

  def do_POST(self):
    if self.path.rstrip('/') != '/v1/chat/completions':
      return self._json(404, {'error': {'message': 'not found'}})
    try:
      body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
      model_id = body['model'].split(',')[0]
      messages = body['messages']
      if not messages or messages[-1].get('role') != 'user':
        raise ValueError('the last message must be from the user')
      history = [{'role': m['role'], 'content': _to_litert_content(m['content'])} for m in messages[:-1]]
      message = {'role': 'user', 'content': _to_litert_content(messages[-1]['content'])}
      engine, lock = self.engines.get(model_id)
    except KeyError as e:
      return self._json(404, {'error': {'message': f'unknown model {e}'}})
    except Exception as e:  # pylint: disable=broad-exception-caught
      return self._json(400, {'error': {'message': repr(e)}})

    sampler = litert_lm.SamplerConfig(temperature=float(body.get('temperature', 0.0)))
    stream = bool(body.get('stream'))
    rid, created = f'chatcmpl-{uuid.uuid4().hex[:12]}', int(time.time())
    t0 = time.time()
    with lock, engine.create_conversation(messages=history or None, sampler_config=sampler) as conv:
      if not stream:
        text = _chunk_text(conv.send_message(message))
        print(f'{model_id}: {time.time() - t0:.2f} s', flush=True)
        return self._json(200, {'id': rid, 'object': 'chat.completion', 'created': created, 'model': model_id,
                                'choices': [{'index': 0, 'finish_reason': 'stop',
                                             'message': {'role': 'assistant', 'content': text}}]})
      self.send_response(200)
      self._cors()
      self.send_header('Content-Type', 'text/event-stream')
      self.send_header('Cache-Control', 'no-cache')
      self.send_header('Connection', 'close')
      self.end_headers()
      self.close_connection = True

      def send(delta, finish=None):
        data = {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id,
                'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
        self.wfile.write(f'data: {json.dumps(data)}\n\n'.encode())
        self.wfile.flush()

      try:
        send({'role': 'assistant'})
        for chunk in conv.send_message_async(message):
          text = _chunk_text(chunk)
          if text:
            send({'content': text})
        send({}, 'stop')
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
        print(f'{model_id}: {time.time() - t0:.2f} s (streamed)', flush=True)
      except (BrokenPipeError, ConnectionResetError):
        conv.cancel_process()  # the page stopped listening
        print(f'{model_id}: client went away after {time.time() - t0:.2f} s', flush=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--port', type=int, default=9379)
  ap.add_argument('--cors-origin', action='append', default=[], help='allowed page origin (repeatable; * for any)')
  ap.add_argument('--models', default='gemma-4-e4b,gemma-4-e2b')
  ap.add_argument('--warm', default='gemma-4-e4b', help='model to load and warm up at start ("" for none)')
  ap.add_argument('--keep_all', action='store_true',
                  help='keep every model loaded once used (needs RAM for all of them)')
  a = ap.parse_args()
  litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
  engines = Engines(a.models.split(','), keep_one=not a.keep_all)
  if not engines.ids:
    sys.exit(f'no models in {MODELS_DIR}; import them with `litert-lm import` (tools/gemma_server.sh does)')
  Handler.engines, Handler.origins = engines, set(a.cors_origin or ['http://localhost:5175'])
  for m in filter(None, a.warm.split(',')):
    if m in engines.ids:
      engines.warm_up(m)
  print(f'Gemma server on http://127.0.0.1:{a.port} (models {", ".join(engines.ids)}; '
        f'origins {", ".join(sorted(Handler.origins))})', flush=True)
  ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
  main()
