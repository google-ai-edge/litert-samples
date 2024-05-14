# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================


# Download yamnet.tflite from the internet if it's not exist.
TFLITE_FILE=./AudioClassification/Services/yamnet.tflite
if test -f "$TFLITE_FILE"; then
    echo "INFO: yamnet.tflite existed. Skip downloading and use the local model."
else
    curl -o ${TFLITE_FILE} https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
    echo "INFO: Downloaded yamnet.tflite to $TFLITE_FILE ."
fi

# Download speech_commands.tflite from the internet if it's not exist.
TFLITE_FILE=./AudioClassification/Services/speech_commands.tflite
if test -f "$TFLITE_FILE"; then
    echo "INFO: speech_commands.tflite existed. Skip downloading and use the local model."
else
    curl -o ${TFLITE_FILE} https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/ios/speech_commands.tflite
    echo "INFO: Downloaded speech_commands.tflite to $TFLITE_FILE ."
fi
