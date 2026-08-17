# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""List of ASR models supported by the conversion script and their configs."""

from samples.litert.speech_recognition.convert import moonshine
from samples.litert.speech_recognition.convert import parakeet_ctc
from samples.litert.speech_recognition.convert import parakeet_tdt
from samples.litert.speech_recognition.convert import qwen3_asr
from samples.litert.speech_recognition.convert import whisper


SUPPORTED_MODELS = {
    parakeet_ctc.ParakeetCTC.HF_MODEL_ID: parakeet_ctc.ParakeetCTC,
    parakeet_tdt.ParakeetTDT.HF_MODEL_ID: parakeet_tdt.ParakeetTDT,
    parakeet_tdt.ParakeetTDTCTCJa.HF_MODEL_ID: parakeet_tdt.ParakeetTDTCTCJa,
    moonshine.Moonshine.HF_MODEL_ID: moonshine.Moonshine,
    whisper.Whisper.HF_MODEL_ID: whisper.Whisper,
    qwen3_asr.Qwen3Asr.HF_MODEL_ID: qwen3_asr.Qwen3Asr,
}
