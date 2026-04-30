import os
import sys

try:
    from ai_edge_litert.aot import aot_compile as aot_lib
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
except ImportError:
    print("Error: 'ai-edge-litert' is not installed.")
    print("Please run: pip install ai-edge-litert")
    sys.exit(1)

# Path to QAIRT SDK provided by the user
os.environ["QAIRT_ROOT"] = "/mnt/c/Users/rawat/Downloads/v2.42.0.251225/qairt/2.42.0.251225"
os.environ["PATH"] = os.environ["QAIRT_ROOT"] + "/bin/aarch64-ubuntu-gcc9.4:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = os.environ["QAIRT_ROOT"] + "/lib/aarch64-ubuntu-gcc9.4:" + os.environ.get("LD_LIBRARY_PATH", "")

def compile_for_npu(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Starting AOT compilation for {model_path}...")
    print("Target: Snapdragon 8 Elite (SM8750)")
    
    sm8750_target = qnn_target.Target(qnn_target.SocModel.SM8750)
    
    try:
        # Create output directory
        out_dir = "compiled"
        os.makedirs(out_dir, exist_ok=True)
        
        # This will generate a compiled version of the model
        compiled_models = aot_lib.aot_compile(
            model_path, 
            output_dir=out_dir,
            target=[sm8750_target]
        )
        print(f"Compilation successful! Output saved in '{out_dir}/'")
        print("The compiled model should now contain the TF_LITE_AUX payload.")
    except Exception as e:
        print(f"Compilation failed: {e}")

if __name__ == "__main__":
    tflite_model_path = "gemma-4-E2B-it.litertlm"
    compile_for_npu(tflite_model_path)
