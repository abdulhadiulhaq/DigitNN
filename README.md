# 🧠 DigitNN — Handwritten Digit Recognizer

<div align="center">

```
██████╗ ██╗ ██████╗ ██╗████████╗███╗   ██╗███╗   ██╗
██╔══██╗██║██╔════╝ ██║╚══██╔══╝████╗  ██║████╗  ██║
██║  ██║██║██║  ███╗██║   ██║   ██╔██╗ ██║██╔██╗ ██║
██║  ██║██║██║   ██║██║   ██║   ██║╚██╗██║██║╚██╗██║
██████╔╝██║╚██████╔╝██║   ██║   ██║ ╚████║██║ ╚████║
╚═════╝ ╚═╝ ╚═════╝ ╚═╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═══╝
```

![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Raylib](https://img.shields.io/badge/Raylib-GUI-black?style=for-the-badge&logo=raylib&logoColor=white)
![MNIST](https://img.shields.io/badge/MNIST-60k%20Samples-orange?style=for-the-badge)
![CMake](https://img.shields.io/badge/CMake-Build-064F8C?style=for-the-badge&logo=cmake&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A neural network built from scratch in C++ that recognizes handwritten digits in real time.**
*No ML libraries. No shortcuts. Pure math.*

</div>

---

## ✨ What It Does

Draw a digit with your mouse → the network predicts it **instantly**.

```
┌─────────────────────────────┬──────────────┐
│                             │  Draw a digit│
│   You draw here (28×28)     │              │
│                             │      5       │ ← predicted
│   Left click  = draw        │              │
│   Right click = clear       │   [Clear]    │
└─────────────────────────────┴──────────────┘
```

---

## 🏗️ Architecture

```
INPUT         HIDDEN 1       HIDDEN 2       OUTPUT
784 neurons → 128 neurons → 64 neurons  → 10 neurons
(28×28 px)    (ReLU)         (ReLU)        (Sigmoid)
                                           0 1 2 3 4 5 6 7 8 9
```

Everything is **hand-coded from scratch**:
- Matrix class with multiply, transpose, add, subtract
- Forward pass
- Backpropagation with chain rule
- SGD weight updates
- MNIST binary file parser
- Raylib drawing canvas

---

## 📁 Project Structure

```
DigitNN/
├── include/
│   ├── Matrix.h          # Matrix class (NN namespace)
│   ├── Layer.h           # Single dense layer
│   ├── Network.h         # Full network (forward + backprop)
│   └── DataLoader.h      # MNIST binary parser
├── src/
│   ├── Matrix.cpp        # Linear algebra implementation
│   ├── Layer.cpp         # ReLU + Sigmoid activations
│   ├── Network.cpp       # Training loop + weight save/load
│   └── DataLoader.cpp    # IDX file format parser
├── main.cpp              # Training entry point
├── canvas_test.cpp       # Raylib GUI entry point
├── CMakeLists.txt        # Build configuration
└── data/
    ├── train-images.idx3-ubyte
    ├── train-labels.idx1-ubyte
    ├── t10k-images.idx3-ubyte
    ├── t10k-labels.idx1-ubyte
    └── weights.txt       # saved after training
```

---

## ⚙️ How It Works

### Forward Pass
```
Z = W · A_prev + b      ← linear transformation
A = activation(Z)       ← ReLU for hidden, Sigmoid for output
```

### Backpropagation
```
Output delta  = (A_last - expected) ⊙ A_last*(1 - A_last)
Hidden delta  = (W_next^T · delta)  ⊙ ReLU'(Z)
Weight update = W - lr * delta * A_prev^T
Bias update   = B - lr * delta
```

### Why These Choices?

| Choice | Why |
|--------|-----|
| ReLU hidden | Fast, no vanishing gradient in positive range |
| Sigmoid output | Squashes to (0,1), interpretable as confidence |
| MSE loss | Simple, integrates cleanly with sigmoid derivative |
| SGD | 60,000 weight updates per epoch, fast convergence |
| Weights (-0.1, 0.1) | Prevents saturation at start of training |

---

## 🚀 Getting Started

### Prerequisites
- CMake 3.10+
- MinGW / GCC with C++17
- [Raylib](https://www.raylib.com/) installed
- MNIST dataset files in `data/`

### Build
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Train
```bash
./Train.exe
# Trains for 50 epochs, saves weights.txt
# Watch loss decrease each epoch
```

### Run
```bash
./Canvas.exe
# Opens drawing window
# Draw a digit, press SPACE to predict
# Right click to clear
```

---

## 📈 Training Progress

Loss decreases each epoch as the network learns:

```
Epoch  1 │████████████████████░░░░░░░░░░│ Loss: 0.118
Epoch  5 │████████████████████████░░░░░░│ Loss: 0.071
Epoch 10 │██████████████████████████░░░░│ Loss: 0.048
Epoch 20 │████████████████████████████░░│ Loss: 0.031
Epoch 50 │██████████████████████████████│ Loss: 0.018
```

---

## 🧩 Key Implementation Details

### The NN Namespace
Raylib defines its own `Matrix` (4×4 float for 3D transforms). To avoid collision, our Matrix lives inside `namespace NN`. In `canvas_test.cpp` we use `NN::Matrix` explicitly.

### Why Z and A Are Stored Separately
Backprop needs both:
- `Z` (pre-activation) → ReLU derivative: `Z > 0 ? 1 : 0`
- `A` (post-activation) → Sigmoid derivative: `A * (1 - A)`, weight gradients

### The Assignment Operator Bug (Fixed)
Early version forgot to resize `data` in `operator=`. Fixed with:
```cpp
data.assign(number_rows, vector<double>(number_columns, 0.0));
```

---

## 🛠️ Built With

- **C++17** — core language
- **Raylib** — cross-platform GUI
- **MNIST** — 60,000 training images
- **CMake** — build system

---

## 📄 License

MIT License — free to use, learn from, and modify.

---

<div align="center">

*Built as a learning project to understand neural networks from the ground up.*
*Every matrix multiply, every gradient, every pixel — written by hand.*

</div>
