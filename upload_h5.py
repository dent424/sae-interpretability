"""Download large file to Modal volume from Google Drive.

Usage:
    py -3.12 -m modal run upload_h5.py
"""
import modal

app = modal.App("upload-data")
volume = modal.Volume.from_name("sae-data", create_if_missing=True)

GDRIVE_FILE_ID = "1erdgfxdGCx8kFR1qdp16EJQe2ilV_Yh3"
H5_FILENAME = "mexican_national_sae_features_e32_k32_lr0_0003-final.h5"

image = modal.Image.debian_slim().pip_install("gdown")

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,  # 1 hour
)
def download_h5():
    """Download H5 file from Google Drive to Modal volume."""
    import gdown
    import os

    dest_path = f"/data/{H5_FILENAME}"

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {H5_FILENAME} from Google Drive...")

    # Download to /tmp first (recommended by Modal)
    tmp_path = f"/tmp/{H5_FILENAME}"
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, tmp_path, quiet=False)

    # Move to volume
    import shutil
    shutil.move(tmp_path, dest_path)

    # Commit the volume changes
    volume.commit()

    size_gb = os.path.getsize(dest_path) / 1e9
    print(f"Done! File saved to {dest_path} ({size_gb:.2f} GB)")

@app.local_entrypoint()
def main():
    download_h5.remote()
