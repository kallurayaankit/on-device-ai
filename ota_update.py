import requests
import time
import os

OTA_URL = "http://localhost:8000"
MODEL_DIR = "models"
CURRENT_VERSION_FILE = "current_version.txt"

def get_current_version():
    """Read the version we already have (or assume version 1)."""
    if os.path.exists(CURRENT_VERSION_FILE):
        with open(CURRENT_VERSION_FILE, "r") as f:
            return f.read().strip()
    else:
        return "1"   # initial version before any update

def save_current_version(version):
    with open(CURRENT_VERSION_FILE, "w") as f:
        f.write(version)

def check_for_update():
    """Ask the server what the latest version is."""
    try:
        response = requests.get(f"{OTA_URL}/model_version.txt")
        if response.status_code == 200:
            return response.text.strip()
    except Exception:
        print("Could not reach OTA server.")
    return None

def download_model(version):
    """Download the new model from the server."""
    url = f"{OTA_URL}/model_v{version}.onnx"
    local_path = os.path.join(MODEL_DIR, f"model_v{version}.onnx")
    print(f"Downloading model version {version}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        print(f"Model saved as {local_path}")
        return local_path
    else:
        print(f"Failed to download version {version}")
        return None

# Main loop: check every 10 seconds
print("OTA client started. Checking for updates every 10 seconds...")
while True:
    current_version = get_current_version()
    latest_version = check_for_update()

    if latest_version and latest_version != current_version:
        print(f"New version available: {latest_version} (we have {current_version})")
        new_model = download_model(latest_version)
        if new_model:
            save_current_version(latest_version)
            print("Model updated successfully!")
            # In a real app you would reload the model here.
    else:
        print(f"No new update. Current version: {current_version}")

    time.sleep(10)