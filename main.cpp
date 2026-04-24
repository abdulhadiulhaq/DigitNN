#include "Network.h"
#include "DataLoader.h"
#include <iostream>
#include <ctime>
using namespace std;

// oneHot
// Converts an integer label (0-9) into a (10×1) one-hot column vector.
// One-hot encoding: all zeros except a single 1.0 at the index of the label.
//
// Example: label=5 → [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]^T
//
// Why one-hot?
//   The network outputs 10 sigmoid values, each representing confidence for
//   one digit. We want the correct digit's neuron to output 1.0 and all
//   others to output 0.0. One-hot gives us exactly that as the target.
NN::Matrix oneHot(int label) {
    NN::Matrix m(10, 1);
    m.set_element(label, 0, 1.0);
    return m;
}

// loss — Mean Squared Error (MSE)
// Measures how wrong the network's prediction is compared to the expected output.
//
// Formula: MSE = (1/n) * Σ(expected[i] - actual[i])²
//
// For each of the 10 output neurons, we square the difference between the
// expected value (0 or 1) and the actual sigmoid output (0.0 to 1.0).
// Squaring serves two purposes:
//   1. Makes all errors positive (a negative error is still bad)
//   2. Penalizes large errors more heavily than small ones
//
// We average over n=10 neurons to get a normalized loss value.
// Perfect prediction = loss 0.0, worst case ≈ loss 1.0.
double loss(const NN::Matrix& expected, const NN::Matrix& actual) {
    double total = 0.0;
    for (int i = 0; i < expected.get_row(); i++) {
        double val = expected.get_matrix_element(i, 0) - actual.get_matrix_element(i, 0);
        total += val * val;
    }
    return total / expected.get_row();
}

int main() {
    // HYPERPARAMETERS
    // epochs:       how many times to iterate over the full 60,000 training set
    //               More epochs = more learning, but diminishing returns after ~20
    // learningRate: how large a step to take in the direction of the gradient
    //               Too high → overshoots, loss bounces or diverges
    //               Too low  → learns very slowly
    //               0.001 is conservative but stable for this architecture
    int epochs = 50;

    // NETWORK ARCHITECTURE {784, 128, 64, 10}
    // Input layer:   784 neurons (one per pixel of 28×28 image)
    // Hidden layer 1: 128 neurons with ReLU activation
    // Hidden layer 2:  64 neurons with ReLU activation
    // Output layer:    10 neurons with Sigmoid activation (one per digit 0-9)
    Network net({784, 128, 64, 10});
    double lr = 0.0001;

    // Load training data from MNIST binary files
    // Path is relative to the build/ directory: ../data/ = DigitNN/data/
    vector<NN::Matrix> images = loadImages("../data/train-images.idx3-ubyte");
    vector<int> labels = loadLabels("../data/train-labels.idx1-ubyte");

    // TRAINING LOOP
    // Outer loop: epochs — each epoch sees all 60,000 samples
    // Inner loop: samples — each sample does one forward pass + one backprop
    //
    // This is Stochastic Gradient Descent (SGD): we update weights after EVERY
    // single sample rather than averaging gradients over a batch first.
    // SGD is noisier than batch gradient descent but converges faster in practice
    // for large datasets because it makes 60,000 updates per epoch instead of 1.
    for (int e = 0; e < epochs; e++) {
        double totalLoss = 0.0;
        time_t start = time(0);
        for (int i = 0; i < 60000; i++) {
            NN::Matrix expected = oneHot(labels[i]);
            NN::Matrix output = net.forward(images[i]);
            totalLoss += loss(expected, output);
            net.train(images[i], expected, lr);
            if (i % 1000 == 0 && i > 0) {
                cout << "Epoch " << e+1 << " Sample " << i << " Loss: " << totalLoss/i << "\n";
            }
        }
        cout << "Epoch " << e+1 << " complete. Avg Loss: " << totalLoss/60000
             << " Time: " << time(0)-start << "s\n";
    }

    // Save trained weights to file so Canvas.exe can load them without retraining
    net.saveWeights("../data/weights.txt");
    cout << "Weights saved.\n";
    return 0;
}