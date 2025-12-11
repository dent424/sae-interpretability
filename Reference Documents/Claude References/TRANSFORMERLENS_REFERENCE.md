# TransformerLens Reference

Reference for TransformerLens - a library for mechanistic interpretability of transformer models.

## What is TransformerLens?

TransformerLens provides tools to:
- Load transformer models with custom hooks for activation extraction
- Intercept activations at any point in the forward pass
- Extract residual stream, attention, and MLP activations
- Perform surgical interventions on model components

**Primary use:** Extracting activations for SAE training and analysis.

---

## Installation

```bash
pip install transformer-lens
```

---

## Loading Models

### HookedTransformer

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "gpt2",           # Model name
    device="cuda",    # "cuda" or "cpu"
    dtype="float32"   # "float32", "float16", "bfloat16"
)
```

### GPT-2 Model Variants

| Model | D_MODEL | N_LAYERS | N_HEADS |
|-------|---------|----------|---------|
| gpt2 | 768 | 12 | 12 |
| gpt2-medium | 1024 | 24 | 16 |
| gpt2-large | 1280 | 36 | 20 |
| gpt2-xl | 1600 | 48 | 25 |

---

## Hook System

### Basic Pattern

```python
# Storage for activations
all_activations = []

# Define hook function
def save_acts(act, hook):
    """
    act: Activation tensor at hook point
    hook: Hook metadata
    """
    all_activations.append(act.detach())

# Run with hooks
model.run_with_hooks(
    input_ids,                      # Token tensor
    fwd_hooks=[(hook_name, save_acts)],  # List of (name, function) tuples
    return_type=None                # None = don't return model output
)
```

### Hook Function Signature

```python
def hook_fn(activation: torch.Tensor, hook) -> Optional[torch.Tensor]:
    """
    Args:
        activation: The tensor at this hook point
        hook: Hook object with metadata

    Returns:
        None to leave activation unchanged
        Modified tensor to intervene
    """
    # For extraction: just save
    saved.append(activation.detach())

    # For intervention: return modified tensor
    # return activation * 0.5
```

---

## Hook Names

### Residual Stream

```python
f"blocks.{layer}.hook_resid_pre"   # BEFORE layer computation (input)
f"blocks.{layer}.hook_resid_post"  # AFTER layer computation (output)
```

**For SAE work, typically use `hook_resid_pre`** - this captures the residual stream as it enters a layer.

### Attention Block

```python
f"blocks.{layer}.attn.hook_q"      # Query vectors
f"blocks.{layer}.attn.hook_k"      # Key vectors
f"blocks.{layer}.attn.hook_v"      # Value vectors
f"blocks.{layer}.attn.hook_attn"   # Attention weights (softmax output)
f"blocks.{layer}.attn.hook_out"    # Attention output
```

### MLP Block

```python
f"blocks.{layer}.mlp.hook_in"      # MLP input
f"blocks.{layer}.mlp.hook_pre"     # Pre-activation (before GELU)
f"blocks.{layer}.mlp.hook_out"     # MLP output
```

### Layer Index

For GPT-2: layers 0-11 (12 total)

```python
LAYER_IDX = 8
HOOK = f"blocks.{LAYER_IDX}.hook_resid_pre"  # Layer 8 residual input
```

---

## Complete Activation Extraction Pattern

This is the pattern used in the SAE interpretability project:

```python
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# =============================================================================
# SETUP
# =============================================================================

model = HookedTransformer.from_pretrained("gpt2", device="cuda", dtype="float32")
tok = AutoTokenizer.from_pretrained("gpt2")

LAYER_IDX = 8
HOOK = f"blocks.{LAYER_IDX}.hook_resid_pre"
D_MODEL = 768

# =============================================================================
# TOKENIZE
# =============================================================================

text = "Your text here"
ids = tok(text, add_special_tokens=False).input_ids

# Pad to create 64-token windows (matching SAE training)
windows = []
for i in range(0, len(ids), 64):
    window = ids[i:i+64]
    if len(window) < 64:
        window = window + [tok.eos_token_id] * (64 - len(window))
    windows.append(window)

windows_tensor = torch.tensor(windows).cuda()

# =============================================================================
# EXTRACT ACTIVATIONS
# =============================================================================

all_activations = []

def save_acts(act, hook):
    """Extract and normalize activations"""
    flat = act.detach().reshape(-1, D_MODEL)

    # Normalize: mean-center and L2-normalize
    flat = flat - flat.mean(-1, keepdim=True)
    flat = flat / flat.norm(dim=-1, keepdim=True)

    all_activations.append(flat)

model.run_with_hooks(
    windows_tensor,
    fwd_hooks=[(HOOK, save_acts)],
    return_type=None
)

# Concatenate: shape (n_tokens, D_MODEL)
gpt2_acts = torch.cat(all_activations, dim=0)

# =============================================================================
# PASS THROUGH SAE
# =============================================================================

with torch.no_grad():
    x_hat, z = sae(gpt2_acts)  # z = sparse feature activations
```

---

## Multiple Hooks

Extract from multiple points simultaneously:

```python
pre_acts = []
post_acts = []

def save_pre(act, hook):
    pre_acts.append(act.detach())

def save_post(act, hook):
    post_acts.append(act.detach())

model.run_with_hooks(
    input_ids,
    fwd_hooks=[
        ("blocks.8.hook_resid_pre", save_pre),
        ("blocks.8.hook_resid_post", save_post),
    ],
    return_type=None
)
```

---

## Extract All Layers

```python
layer_activations = {}

for layer_idx in range(12):
    activations = []

    def save_acts(act, hook, layer=layer_idx):
        activations.append(act.detach())

    model.run_with_hooks(
        input_ids,
        fwd_hooks=[(f"blocks.{layer_idx}.hook_resid_pre", save_acts)],
        return_type=None
    )

    layer_activations[layer_idx] = torch.cat(activations, dim=0)
```

---

## Intervention (Activation Patching)

Modify activations during forward pass:

```python
def zero_feature_hook(act, hook):
    """Zero out a specific feature direction"""
    # act shape: (batch, seq, d_model)
    act[..., 42] = 0  # Zero dimension 42
    return act  # Return modified activation

logits = model.run_with_hooks(
    input_ids,
    fwd_hooks=[("blocks.8.hook_resid_pre", zero_feature_hook)],
    return_type="logits"  # Get model output
)
```

---

## Activation Shapes

| Hook | Shape | Description |
|------|-------|-------------|
| `hook_resid_pre/post` | (batch, seq, d_model) | Residual stream |
| `attn.hook_q/k/v` | (batch, seq, n_heads, d_head) | Attention vectors |
| `attn.hook_attn` | (batch, n_heads, seq, seq) | Attention weights |
| `mlp.hook_out` | (batch, seq, d_model) | MLP output |

For GPT-2: d_model=768, n_heads=12, d_head=64

---

## Common Gotchas

1. **Always `.detach()`** - Prevents gradient accumulation
2. **Reshape for SAE** - SAE expects (n_tokens, d_model), not (batch, seq, d_model)
3. **return_type=None** - When you only want to extract, not generate
4. **Window padding** - Match the window size used in SAE training (64 tokens)
5. **Normalization** - Apply same normalization as SAE training

---

## Resources

- [GitHub: TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [Documentation](https://transformerlensorg.github.io/)
- [Mechanistic Interpretability Glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)
