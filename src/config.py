"""SAE configuration constants."""

# Model dimensions
D_MODEL = 768              # GPT-2 hidden dimension
EXPANSION = 32             # SAE expansion factor
K_ACTIVE = 32              # Top-K sparsity (active features per token)
N_LATENTS = 24576          # Total features (D_MODEL * EXPANSION)

# Hook configuration
LAYER_IDX = 8
HOOK = f"blocks.{LAYER_IDX}.hook_resid_pre"

# Processing
WINDOW_SIZE = 64           # Token window size for processing

# Modal Volume paths (when mounted at /data)
VOLUME_MOUNT = "/data"
H5_PATH = f"{VOLUME_MOUNT}/mexican_national_sae_features_e32_k32_lr0_0003-final.h5"
METADATA_PATH = f"{VOLUME_MOUNT}/mexican_national_metadata.npz"
SAE_CHECKPOINT_PATH = f"{VOLUME_MOUNT}/sae_e32_k32_lr0.0003-final.pt"
