

# Download yamnet.tflite from the internet if it's not exist.
TFLITE_FILE=./AudioClassification/Services/yamnet.tflite
if test -f "$TFLITE_FILE"; then
    echo "INFO: yamnet.tflite existed. Skip downloading and use the local model."
else
    curl -o ${TFLITE_FILE} https://storage.googleapis.com/ai-edge/interpreter-samples/audio_classification/ios/yamnet.tflite
    echo "INFO: Downloaded yamnet.tflite to $TFLITE_FILE ."
fi

# Download speech_commands.tflite from the internet if it's not exist.
TFLITE_FILE=./AudioClassification/Services/speech_commands.tflite
if test -f "$TFLITE_FILE"; then
    echo "INFO: speech_commands.tflite existed. Skip downloading and use the local model."
else
    curl -o ${TFLITE_FILE} https://storage.googleapis.com/ai-edge/interpreter-samples/audio_classification/ios/speech_commands.tflite
    echo "INFO: Downloaded speech_commands.tflite to $TFLITE_FILE ."
fi
