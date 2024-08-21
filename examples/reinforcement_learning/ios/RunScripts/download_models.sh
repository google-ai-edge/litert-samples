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

# Download planestrike.tflite from the internet if it's not exist.
MODEL_FILE=./ReinforcementLearning/Agent/planestrike_tf.tflite
if test -f "$MODEL_FILE"; then
    echo "INFO: planestrike_tf.tflite existed. Skip downloading and use the local task."
else
    curl -o ${MODEL_FILE} https://storage.googleapis.com/ai-edge/interpreter-samples/reinforcement_learning/ios/planestrike_tf.tflite -L
    echo "INFO: Downloaded planestrike.tflite to $MODEL_FILE ."
fi
