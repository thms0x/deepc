# deepc: Minimalist C Deep Learning Framework

`deepc` is a lightweight, graph-based deep learning framework implemented in C. It is designed to be simple, educational, and dependency-free (relying only on the standard C library and math functions).

## Credits

This project is based on the following video tutorial:
- [Coding a machine learning library in C from scratch](https://www.youtube.com/watch?v=hL_n_GljC0I&t=1s)

The framework features a custom matrix library, a modular computation graph system, and an example implementation for training an MNIST digit classifier.

## Features

- **Matrix Library:** Comprehensive linear algebra operations (addition, subtraction, multiplication, scaling, transposition).
- **Cache-Optimized Matmul:** Matrix multiplication is optimized for cache locality using loop reordering (e.g., `i-k-j` order for standard multiplication) to ensure contiguous memory access and improve performance.
- **Computation Graph:** Graph-based model definition with automatic differentiation support for backpropagation.
- **Activation Functions:** Optimized implementations of ReLU and Softmax.
- **Loss Functions:** Cross-Entropy Loss for classification tasks.
- **Modular Layers:** Easily extensible layer system, starting with Fully Connected (Dense) layers.
- **MNIST Example:** A complete end-to-end example for training a neural network on the MNIST dataset.

## Project Structure

```text
.
├── include/          # Header files
│   ├── matrix.h      # Matrix operations and definitions
│   ├── model.h       # Computation graph and model management
│   └── layer.h       # Layer implementations (e.g., Fully Connected)
├── src/              # Source files
│   ├── matrix.c
│   ├── model.c
│   ├── layer.c
│   └── main.c        # MNIST training demonstration
├── data/             # Dataset directory (created by mnist.py)
├── Makefile          # Build system
├── mnist.py          # Data preparation script
└── README.md         # This file
```

## Getting Started

### Prerequisites

- A C compiler (e.g., `gcc`)
- `make` build tool
- Python 3 with `numpy` and `torchvision` (only for downloading and preparing the MNIST dataset)

### Data Preparation

The MNIST dataset needs to be downloaded and converted into a binary format that the C program can read. Run the provided Python script:

```bash
python3 mnist.py
```

This will create the `data/` directory and populate it with `.mat` binary files.

### Compilation

Build the project using the provided `Makefile`:

```bash
make
```

This will create a `build/` directory for object files and an executable named `mnist_trainer.out`.

### Running

Execute the trainer:

```bash
./mnist_trainer.out
```

The program will:
1. Load the MNIST training and test datasets.
2. Define a multi-layer perceptron (MLP).
3. Train the model for 1 epoch.
4. Evaluate the model on the test set and report the accuracy.

## Usage Example

Models in `deepc` are built by defining a computation graph within a `model_context`:

```c
void create_model(model_context *model) {
    // Define input variable (784 features for MNIST)
    model_var *input = mv_create(model, 784, 1, MV_FLAG_INPUT);

    // Layer 1: Fully Connected (128 units) + ReLU
    model_var *z0 = layer_fully_connected(model, input, 784, 128);
    model_var *a0 = mv_relu(model, z0, 0);

    // Layer 2: Fully Connected (10 units for digits 0-9)
    model_var *z1 = layer_fully_connected(model, a0, 128, 10);

    // Output: Softmax activation
    model_var *output = mv_softmax(model, z1, MV_FLAG_OUTPUT);

    // Define training targets and cost function
    model_var *y = mv_create(model, 10, 1, MV_FLAG_DESIRED_OUPUT);
    model_var *cost = mv_cross_entropy(model, output, y, MV_FLAG_COST);
}
```

## License

This project is open-source and available under the MIT License.
