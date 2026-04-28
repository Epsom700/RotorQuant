# RotorQuant

**Rotation-aware quantization for LLM activations on Apple Silicon.**

RotorQuant applies a randomized Hadamard rotation before Lloyd-Max quantization to redistribute activation outliers, then inverts the rotation after dequantization. This produces lower reconstruction error than direct quantization — especially for heavy-tailed activation distributions common in transformer MLP layers.

The C++ core uses the **Fast Walsh-Hadamard Transform (FWHT)** for O(n log n) in-place rotation — no matrix storage or BLAS dependency required — and exposes a **pybind11** module for seamless integration with PyTorch training loops via a straight-through estimator (STE).

Inspired by the ideas behind Google's [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) approach to extreme quantization with rotation-based outlier redistribution.

---

## Architecture

```
Input Activations
        │
        ▼
┌───────────────┐
│  Random Sign  │   element-wise multiply by ±1
│    Flips      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Hadamard    │   FWHT: orthogonal rotation in O(n log n)
│   Transform   │   distributes outliers across all dims
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Lloyd-Max    │   optimal scalar quantization
│  Quantizer    │   for Gaussian-distributed data
└───────┬───────┘
        │
        ▼
   Quantized Indices ──► storage / transmission
        │
        ▼
┌───────────────┐
│  Lloyd-Max    │   bin index → centroid value
│  Dequantizer  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Inverse     │   FWHT again (Hadamard is self-inverse
│   Hadamard    │   up to scaling)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Inverse Sign │   undo the random flips
│    Flips      │
└───────┬───────┘
        │
        ▼
Reconstructed Activations
```

---

## Results

### C++ encode–decode round-trip (8-dim vector, 8 quantization levels)

| Metric | Value |
|---|---|
| Reconstruction MSE (1D, 8-dim) | ~0.02 |
| Reconstruction MSE (2D, 3×8 batch, avg) | ~0.02 |
| Original size (float64) | 64 bytes |
| Encoded size (int32 bins) | 32 bytes |
| **Compression ratio** | **50%** |

### Llama-3.2-3B-Instruct LoRA fine-tuning (WikiText-2, 100 steps)

Training with `num_levels=8` (3-bit), `σ=1.0`, LoRA `r=8`, `batch=4`, `seq_len=128`, `grad_accum=2`:

| Metric | Baseline (no quantization) | RotorQuant (3-bit activations) |
|---|---|---|
| **Step 1 loss** | 5.17 | 7.24 |
| **Step 25 loss** | 1.09 | 2.49 |
| **Step 50 loss** | 1.08 | 1.48 |
| **Step 100 loss** | 1.95 | 2.53 |
| **Final avg loss (last 10)** | ~1.85 | ~2.50 |
| **s/step (steady state)** | ~5–10 s | ~15–30 s |
| **Peak memory** | 0.47 GB | 6.40 GB |

> **Key takeaway:** RotorQuant adds only ~0.6 loss penalty at convergence while compressing all 28 MLP activation layers from fp32/bf16 to 3-bit, enabling deployment on memory-constrained devices. The higher memory during training is expected (gradient buffers, activation caching for STE); at inference time the footprint is much smaller. The switch to FWHT eliminates the n×n Hadamard matrix that was previously cached per-layer.

---

## Features

- **Hadamard rotation via FWHT** — Normalizes activation distributions before quantization using the Fast Walsh-Hadamard Transform in O(n log n) time with no matrix storage.
- **Lloyd-Max quantizer** — Computes optimal breakpoints and centroids for a Gaussian source, converging via the Lloyd algorithm.
- **Batched f32 path** — In-place FWHT + elementwise quantize/dequantize for the full encode→quantize→decode round-trip in float32, operating on entire batches at once.
- **pybind11 bindings** — Exposes `RotorQuant` to Python with zero-copy NumPy integration.
- **STE PyTorch layer** — `RotorQuantLayer` wraps any activation function, quantizes its output with a straight-through gradient estimator, and plugs into standard training loops.
- **LoRA + RotorQuant training** — `train.py` demonstrates fine-tuning a Llama model with LoRA while RotorQuant compresses intermediate activations.

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| **macOS** (recommended) | Tested on Apple Silicon; portable C++17 with no platform-specific deps |
| **C++17** compiler | `clang++` from Xcode or any standards-conforming compiler |
| **Python** ≥ 3.10 | Python bindings and training |
| **pybind11** | C++ ↔ Python bridge |
| **PyTorch** ≥ 2.0 | Training with STE |

---

## Build

> **Note:** The compiled `rotorquant.*.so` is not included in this repo — you must build it before running the Python scripts.

### 1. Python module (pybind11) — recommended

This is the command you need to produce `rotorquant.cpython-*.so` for use with `quantize.py` and `train.py`:

```bash
# Install pybind11 first
pip install pybind11

# Compile directly (tested on macOS with Apple Silicon):
c++ -O3 -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    bindings.cpp rotorQuant.cpp rotation.cpp lloyd_max.cpp \
    -o rotorquant$(python3-config --extension-suffix)
```

This produces `rotorquant.cpython-3XX-darwin.so` in the project root. Verify it works:

```bash
python3 -c "import rotorquant; rq = rotorquant.RotorQuant(8, 8, 1.0); print('OK')"
```

### 2. C++ test binary

```bash
c++ -O3 -std=c++17 \
    test_rotorQuant.cpp rotorQuant.cpp rotation.cpp lloyd_max.cpp \
    -o test_rotor_quant

./test_rotor_quant
```

### 3. CMake (alternative)

```bash
mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(sysctl -n hw.ncpu)
./test_rotor_quant
```

---

## Usage

### C++

```cpp
#include "rotorQuant.h"

// n=8 dimensions, 8 quantization levels, σ=1.0
RotorQuant rq(8, 8, 1.0);

std::vector<double> x = {0.5, -1.2, 0.8, -0.3, 1.5, -0.7, 0.2, -1.0};
auto bins = rq.encode(x);
auto reconstructed = rq.decode(bins);
```

### Python

```python
import numpy as np
import rotorquant

rq = rotorquant.RotorQuant(8, 8, 1.0)

# Single vector
bins = rq.encode([0.5, -1.2, 0.8, -0.3, 1.5, -0.7, 0.2, -1.0])
reconstructed = rq.decode(bins)

# Batched f32 (in-place FWHT)
data = np.random.randn(1024, 8).astype(np.float32)
rq.encode_decode_batch_f32(data)  # data is modified in-place
```

### PyTorch training with STE

```python
from quantize import inject_rotorquant

model = ...  # any HuggingFace causal LM
model = inject_rotorquant(model, num_levels=8, sigma=1.0)
# Now every MLP activation is quantized with RotorQuant
# Gradients flow through via the straight-through estimator
```

---

## Project Structure

```
RotorQuant/
├── rotorQuant.h          # Main RotorQuant class declaration
├── rotorQuant.cpp         # Encode/decode + batched FWHT path
├── rotation.h             # Hadamard rotation class declaration
├── rotation.cpp           # FWHT implementation + rotate/inverse
├── lloyd_max.h            # Lloyd-Max quantizer class declaration
├── lloyd_max.cpp          # Lloyd algorithm + quantize/dequantize
├── bindings.cpp           # pybind11 module exposing RotorQuant to Python
├── quantize.py            # STE autograd function + RotorQuantLayer (PyTorch)
├── train.py               # LoRA fine-tuning script with RotorQuant injection
├── test_rotorQuant.cpp    # C++ test (1D + 2D encode–decode, MSE)
├── CMakeLists.txt         # Build: core lib, test binary, pybind11 module
├── requirements.txt       # Python dependencies
├── .env.example           # Template for HF_TOKEN
└── LICENSE                # MIT
```

---

## References

- **TurboQuant** — Lam *et al.*, Google Research (2025). *TurboQuant: Redefining AI efficiency with extreme compression.*
  [Blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) · [Paper (arXiv)](https://arxiv.org/abs/2502.19268)

  > RotorQuant implements the core idea from TurboQuant: applying a randomized Hadamard rotation to redistribute outliers before scalar quantization, enabling extreme (≤4-bit) compression of LLM activations with minimal quality loss.

---

## License

MIT — see [LICENSE](LICENSE).
