# RotorQuant

**Rotation-aware quantization for LLM activations on Apple Silicon.**

RotorQuant applies a randomized Hadamard rotation before Lloyd-Max quantization to redistribute activation outliers, then inverts the rotation after dequantization. This produces lower reconstruction error than direct quantization — especially for heavy-tailed activation distributions common in transformer MLP layers.

The C++ core uses Apple's **Accelerate** framework (`cblas_sgemm` on AMX/NEON) for batched float32 encode–decode, and exposes a **pybind11** module for seamless integration with PyTorch training loops via a straight-through estimator (STE).

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
│   Hadamard    │   orthogonal rotation (H · x)
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
│   Inverse     │   H^T · x (Hadamard is self-inverse
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

## Features

- **Hadamard rotation** — Normalizes activation distributions before quantization to minimize reconstruction MSE.
- **Lloyd-Max quantizer** — Computes optimal breakpoints and centroids for a Gaussian source, converging via the Lloyd algorithm.
- **Batched f32 path** — Uses `cblas_sgemm` (Apple Accelerate / AMX) for the full encode→quantize→decode round-trip in float32, operating on entire batches at once.
- **pybind11 bindings** — Exposes `RotorQuant` to Python with zero-copy NumPy integration.
- **STE PyTorch layer** — `RotorQuantLayer` wraps any activation function, quantizes its output with a straight-through gradient estimator, and plugs into standard training loops.
- **LoRA + RotorQuant training** — `train.py` demonstrates fine-tuning a Llama model with LoRA while RotorQuant compresses intermediate activations.

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| **macOS** with Xcode CLT | Apple Accelerate framework (`cblas_sgemm`) |
| **CMake** ≥ 3.10 | Build system |
| **C++17** compiler | `clang++` from Xcode |
| **Python** ≥ 3.10 | Python bindings and training |
| **pybind11** | C++ ↔ Python bridge |
| **PyTorch** ≥ 2.0 | Training with STE |

---

## Build

### C++ test binary

```bash
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
./test_rotor_quant
```

### Python module (pybind11)

```bash
# Inside a virtual environment with pybind11 installed:
pip install pybind11

mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(sysctl -n hw.ncpu)

# Copy the .so to the project root:
cp rotorquant*.so ..
```

Or compile directly:

```bash
c++ -O3 -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    bindings.cpp rotorQuant.cpp rotation.cpp lloyd_max.cpp \
    -framework Accelerate \
    -o rotorquant$(python3-config --extension-suffix)
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

# Batched f32 (in-place, uses Accelerate)
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
├── rotorQuant.cpp         # Encode/decode + batched Accelerate path
├── rotation.h             # Hadamard rotation class declaration
├── rotation.cpp           # Hadamard matrix generation + rotate/inverse
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

## License

MIT — see [LICENSE](LICENSE).
