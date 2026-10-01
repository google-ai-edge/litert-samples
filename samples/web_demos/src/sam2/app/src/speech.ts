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

// Voice input for "Ask Gemma": Chrome's built-in speech recognition (Web
// Speech API). On-device recognition is used when Chrome has (or can install)
// the language pack (`processLocally`, Chrome 139+); otherwise Chrome's
// server recognition is used, and the caller is told which.

interface RecognitionResult {isFinal: boolean; 0: {transcript: string}}
interface RecognitionEvent {resultIndex: number; results: ArrayLike<RecognitionResult>}
interface Recognition {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
  processLocally?: boolean;
  onresult: ((ev: RecognitionEvent) => void) | null;
  onerror: ((ev: {error: string; message?: string}) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
type Availability = 'available' | 'downloadable' | 'downloading' | 'unavailable';
interface RecognitionCtor {
  new(): Recognition;
  available?(o: {langs: string[]; processLocally: boolean}): Promise<Availability>;
  install?(o: {langs: string[]; processLocally: boolean}): Promise<boolean>;
}

const Ctor = ((window as unknown as {SpeechRecognition?: RecognitionCtor}).SpeechRecognition ??
    (window as unknown as {webkitSpeechRecognition?: RecognitionCtor}).webkitSpeechRecognition);

export const speechSupported = () => !!Ctor;

export type SpeechMode = 'on-device' | 'Chrome server';

export interface DictationHandlers {
  /** Text so far; `final` once the utterance is done. */
  onText(text: string, final: boolean): void;
  /** Status to show, e.g. while the on-device language pack installs. */
  onStatus(text: string): void;
  /** Recognition ended; `error` is set if it failed. */
  onEnd(error?: string): void;
}

/** On-device recognition for `lang`, installing the language pack if Chrome offers one. */
async function onDevice(lang: string, onStatus: (t: string) => void): Promise<boolean> {
  if (!Ctor?.available) return false;
  const o = {langs: [lang], processLocally: true};
  try {
    const a = await Ctor.available(o);
    if (a === 'available') return true;
    if ((a === 'downloadable' || a === 'downloading') && Ctor.install) {
      onStatus(`Installing Chrome's on-device speech model for ${lang}…`);
      return await Ctor.install(o);
    }
  } catch (e) {
    console.warn('on-device speech recognition check failed', e);
  }
  return false;
}

let active: Recognition | null = null;

export const listening = () => !!active;

/** Stops listening; the text heard so far is delivered as final. */
export function stopDictation() {
  active?.stop();
}

/** Listens for one utterance. Resolves with the mode once recognition has started. */
export async function startDictation(h: DictationHandlers, lang = navigator.language || 'en-US'): Promise<SpeechMode> {
  if (!Ctor) throw new Error('this browser has no speech recognition');
  if (active) active.abort();
  const local = await onDevice(lang, h.onStatus);
  const rec = new Ctor();
  rec.lang = lang;
  rec.interimResults = true;
  rec.continuous = false;
  rec.maxAlternatives = 1;
  if (local) rec.processLocally = true;
  let text = '', error: string | undefined;
  rec.onresult = (ev) => {
    let heard = '', final = true;
    for (let i = 0; i < ev.results.length; i++) {
      heard += ev.results[i][0].transcript;
      final &&= ev.results[i].isFinal;
    }
    text = heard.trim();
    h.onText(text, false);
    if (final && text) h.onText(text, true);
  };
  rec.onerror = (ev) => {
    error = ev.error === 'not-allowed' ? 'microphone access was denied'
      : ev.error === 'no-speech' ? 'no speech heard'
      : ev.error === 'network' ? 'Chrome speech recognition needs the network (no on-device model for this language)'
      : ev.error === 'language-not-supported' ? `on-device recognition doesn't support ${lang}`
      : ev.message || ev.error;
  };
  rec.onend = () => {
    if (active === rec) active = null;
    h.onEnd(error);
  };
  active = rec;
  rec.start();
  return local ? 'on-device' : 'Chrome server';
}
