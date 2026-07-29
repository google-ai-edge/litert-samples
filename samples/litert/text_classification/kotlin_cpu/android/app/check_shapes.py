import json
import subprocess

def get_tflite_info(filename):
    print(f"--- Info for {filename} ---")
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=filename)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        for i, detail in enumerate(input_details):
            print(f"Input {i}: shape={detail['shape']}, dtype={detail['dtype']}, name={detail['name']}")
            
        for i, detail in enumerate(output_details):
            print(f"Output {i}: shape={detail['shape']}, dtype={detail['dtype']}, name={detail['name']}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

get_tflite_info("src/main/assets/mobile_bert.tflite")
get_tflite_info("src/main/assets/word_vec.tflite")
