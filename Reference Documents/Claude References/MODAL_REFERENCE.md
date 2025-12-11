# Modal Reference Guide

Quick reference for Modal serverless GPU platform patterns.

## Installation & Setup

```bash
pip install modal
modal setup  # authenticate
```

## Core Concepts

### App Creation

```python
import modal

app = modal.App("my-app-name")
```

### Container Images

Define environments as code (no Dockerfile needed):

```python
# Basic image with pip packages
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch",
    "transformers",
    "numpy",
    "h5py"
)

# With uv (faster package manager)
image = modal.Image.debian_slim().uv_pip_install("transformers[torch]")

# Chained operations
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy")
    .run_commands("apt-get update && apt-get install -y git")
)
```

### Function Decorator

```python
@app.function(
    image=image,
    gpu="H100",
    timeout=1200,
    secrets=[modal.Secret.from_name("my-secret")],
    volumes={"/data": volume}
)
def my_function(arg1, arg2):
    # runs in cloud container
    return result
```

**Common Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `image` | Container environment | `image=my_image` |
| `gpu` | GPU type/count | `gpu="H100"`, `gpu="A100:4"` |
| `cpu` | CPU cores | `cpu=4` |
| `memory` | RAM allocation | `memory=32768` (MB) |
| `timeout` | Max runtime (seconds) | `timeout=3600` |
| `secrets` | Credentials | `secrets=[modal.Secret.from_name("x")]` |
| `volumes` | Persistent storage | `volumes={"/path": vol}` |

### GPU Options

```python
@app.function(gpu="any")       # auto-select cheapest available
@app.function(gpu="T4")        # NVIDIA T4 (budget)
@app.function(gpu="A10G")      # NVIDIA A10G
@app.function(gpu="A100")      # NVIDIA A100
@app.function(gpu="H100")      # NVIDIA H100 (fastest)
@app.function(gpu="H100:4")    # 4x H100 GPUs
```

### Volumes (Persistent Storage)

```python
# Create or reference existing volume
volume = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(volumes={"/data": volume})
def process_data():
    # Read from volume
    with open("/data/input.txt") as f:
        data = f.read()

    # Write to volume
    with open("/data/output.txt", "w") as f:
        f.write(result)

    # IMPORTANT: commit changes
    volume.commit()
```

### Local Entrypoint

Orchestrates remote function calls from local machine:

```python
@app.local_entrypoint()
def main():
    # .remote() runs function in cloud
    result = my_function.remote(arg1, arg2)
    print(result)
```

### Stateful Classes

For loading models once and reusing across calls:

```python
@app.cls(image=image, gpu="T4")
class ModelServer:
    @modal.enter()
    def setup(self):
        # Runs once when container starts
        self.model = load_model()

    @modal.method()
    def predict(self, input_data):
        return self.model(input_data)

# Usage
@app.local_entrypoint()
def main():
    server = ModelServer()
    result = server.predict.remote(data)
```

## Execution

```bash
# Run locally-defined entrypoint
modal run my_script.py

# Run specific function
modal run my_script.py::my_function

# Deploy as persistent endpoint
modal deploy my_script.py
```

## Common Patterns

### Pattern 1: GPU Inference

```python
import modal

app = modal.App("inference")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "transformers"
)

@app.cls(image=image, gpu="T4")
class Inference:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("model-name")
        self.model.cuda()

    @modal.method()
    def run(self, text):
        return self.model(text)

@app.local_entrypoint()
def main():
    inf = Inference()
    result = inf.run.remote("Hello world")
    print(result)
```

### Pattern 2: Batch Processing

```python
@app.function(image=image, gpu="T4")
def process_batch(items: list):
    results = []
    for item in items:
        results.append(process(item))
    return results

@app.local_entrypoint()
def main():
    batches = [items[i:i+100] for i in range(0, len(items), 100)]
    # Process batches in parallel
    results = list(process_batch.map(batches))
```

### Pattern 3: Data Pipeline with Volume

```python
volume = modal.Volume.from_name("data-store", create_if_missing=True)

@app.function(volumes={"/data": volume})
def download_data():
    # Download and save to volume
    data = fetch_from_source()
    save_to_path("/data/dataset.h5", data)
    volume.commit()

@app.function(image=image, gpu="T4", volumes={"/data": volume})
def train():
    # Load from volume
    data = load_from_path("/data/dataset.h5")
    model = train_model(data)
    save_model("/data/model.pt", model)
    volume.commit()
```

### Pattern 4: Secrets Management

```python
# Create secret via CLI: modal secret create my-api-key API_KEY=xxx

@app.function(secrets=[modal.Secret.from_name("my-api-key")])
def call_api():
    import os
    api_key = os.environ["API_KEY"]
    # use api_key
```

## Tips

1. **Cold starts**: First invocation may take longer (container spin-up). Use `@app.cls` for warm containers.

2. **Debugging**: Use `modal run --detach` to run in background, check logs with `modal app logs`.

3. **Cost**: Pay per-second of compute. T4 is cheapest, H100 is fastest but most expensive.

4. **Data transfer**: Large data should live in Volumes, not passed as function arguments.

5. **Timeouts**: Default is 300s. Set higher for long-running tasks.

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://modal.com/docs/examples)
- [Modal Discord](https://discord.gg/modal)
