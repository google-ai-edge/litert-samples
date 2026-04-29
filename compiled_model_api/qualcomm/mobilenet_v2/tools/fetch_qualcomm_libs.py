import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "litert_npu_runtime_libraries"
QAIRT_URL = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip"
QAIRT_CONTENT_DIR = "qairt/2.40.0.251030"
QNN_VERSIONS = [69, 73, 75, 79, 81]
JNI_ARM64_DIR = "src/main/jni/arm64-v8a"

def download(url, dest):
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def main():
    tmp_dir = ROOT / "build" / "tmp_qairt"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / "qairt_sdk.zip"

    if not zip_path.exists():
        download(QAIRT_URL, zip_path)

    print("Extracting libraries...")
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Only extract the files we need to save time/space
        for file in zip_ref.namelist():
            if file.endswith(".so"):
                zip_ref.extract(file, extract_dir)

    source_base = extract_dir / QAIRT_CONTENT_DIR

    for version in QNN_VERSIONS:
        dest_dir = RUNTIME_DIR / f"qualcomm_runtime_v{version}" / JNI_ARM64_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Populating {dest_dir}")

        # libQnnHtp.so
        shutil.copy2(source_base / "lib/aarch64-android/libQnnHtp.so", dest_dir / "libQnnHtp.so")
        # libQnnSystem.so
        shutil.copy2(source_base / "lib/aarch64-android/libQnnSystem.so", dest_dir / "libQnnSystem.so")
        # libQnnHtpV{version}Skel.so
        skel_path = source_base / f"lib/hexagon-v{version}/unsigned/libQnnHtpV{version}Skel.so"
        if skel_path.exists():
            shutil.copy2(skel_path, dest_dir / f"libQnnHtpV{version}Skel.so")
        # libQnnHtpV{version}Stub.so
        stub_path = source_base / f"lib/aarch64-android/libQnnHtpV{version}Stub.so"
        if stub_path.exists():
            shutil.copy2(stub_path, dest_dir / f"libQnnHtpV{version}Stub.so")
        
        # libQnnHtpPrepare.so
        prepare_path = source_base / "lib/aarch64-android/libQnnHtpPrepare.so"
        if prepare_path.exists():
            shutil.copy2(prepare_path, dest_dir / "libQnnHtpPrepare.so")

    print("Done!")

if __name__ == "__main__":
    main()
