"""Download GPT-2 to Modal volume.

Usage:
    py -3.12 -m modal run upload_gpt2.py
"""
import modal

app = modal.App("upload-gpt2")
volume = modal.Volume.from_name("sae-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch",
    "transformer-lens",
    "transformers",
)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=1800,  # 30 min
)
def download_gpt2():
    """Download GPT-2 and save to volume."""
    import torch
    import os

    dest_path = "/data/gpt2_model.pt"

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print("Loading GPT-2 from HuggingFace...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2")

    print(f"Saving to {dest_path}...")
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": model.cfg,
    }, dest_path)

    volume.commit()

    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"Done! Saved {size_mb:.1f} MB")

@app.local_entrypoint()
def main():
    download_gpt2.remote()
