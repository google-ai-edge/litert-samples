#  Copyright 2024 The Google AI Edge Authors. All Rights Reserved.

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================


# Download average_word_classifier.tflite from the internet if it's not exist.
TFLITE_FILE=./TextClassification/average_word_classifier.tflite
if test -f "$TFLITE_FILE"; then
    echo "INFO: average_word_classifier.tflite existed. Skip downloading and use the local model."
else
    curl -o ${TFLITE_FILE} https://storage.googleapis.com/ai-edge/interpreter-samples/text_classification/ios/average_word_classifier.tflite
    echo "INFO: Downloaded average_word_classifier.tflite to $TFLITE_FILE ."
fi

# Download bert_classifier.tflite from the internet if it's not exist.
TFLITE_FILE=./TextClassification/bert_classifier.tflite
if test -f "$TFLITE_FILE"; then
    echo "INFO: bert_classifier.tflite existed. Skip downloading and use the local model."
else
    curl -o ${TFLITE_FILE} https://storage.googleapis.com/ai-edge/interpreter-samples/text_classification/ios/bert_classifier.tflite
    echo "INFO: Downloaded bert_classifier.tflite to $TFLITE_FILE ."
fi
