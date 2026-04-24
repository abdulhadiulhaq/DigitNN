#pragma once
#include "Matrix.h"
#include "Layer.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Network{
    public:

    // input: stores the raw input passed to forward() — kept as a member
    //        so it can be referenced during backprop if needed.
    NN::Matrix input;

    // layers: ordered list of all fully-connected layers.
    // For a Network({784, 128, 64, 10}), this contains 3 layers:
    //   layers[0]: 128 neurons, 784 inputs  → weights: 128×784
    //   layers[1]:  64 neurons, 128 inputs  → weights:  64×128
    //   layers[2]:  10 neurons,  64 inputs  → weights:  10×64
    vector<Layer> layers;

    // Z: pre-activation values for each layer, filled during forward().
    //    Z[i] = weights[i] * A[i-1] + biases[i]   (raw weighted sums)
    //    Needed in backprop to compute ReLU derivative: Z[i] > 0 ? 1 : 0
    //
    // A: post-activation values for each layer, filled during forward().
    //    A[i] = activate(Z[i])
    //    Needed in backprop to compute:
    //      - Sigmoid derivative at output: A[last] * (1 - A[last])
    //      - Weight gradient: delta * A[i-1]^T
    //
    // Both vectors are cleared and rebuilt on every forward() call.
    vector<NN::Matrix> Z;
    vector<NN::Matrix> A;

    // CONSTRUCTOR
    // Takes a vector of layer sizes, e.g. {784, 128, 64, 10}.
    // Creates (sizes.size() - 1) layers, where each layer connects sizes[i-1]
    // inputs to sizes[i] outputs. Weights are randomized immediately.
    Network(vector<int> sizes);

    // ELEMENTWISE_MULTIPLY
    // Multiplies two column vectors element by element.
    // Not a standard matrix operation — used specifically in backprop to apply
    // the chain rule: gradient = propagated_error * activation_derivative
    // Both inputs must be (n × 1) column vectors of the same size.
    NN::Matrix elementwise_multiply( const NN::Matrix& a,  const NN::Matrix& b );
    
    // FORWARD
    // Runs the input through all layers sequentially, storing Z and A at each step.
    // Returns the final output A[last] — a (10×1) vector of sigmoid outputs.
    // The highest value in this vector is the network's predicted digit.
    NN::Matrix forward( const NN::Matrix& input);

    // TRAIN — Forward pass + Backpropagation + Weight update in one call.
    // Steps performed:
    //   1. forward(input)         → fills Z and A
    //   2. Compute output delta   → (A[last] - expected) * sigmoid_derivative
    //   3. Update last layer      → W -= lr * delta * A[last-1]^T
    //   4. Loop backwards         → propagate delta through each hidden layer
    //   5. Update each layer      → W -= lr * delta * prev_A^T
    void train(const NN::Matrix& input, const NN::Matrix& expected, double learningRate);
    
    // SAVEWEIGHTS / LOADWEIGHTS
    // Serializes all weights and biases to a plain text file, one value per line.
    // Format: for each layer, all weight values row by row, then all bias values.
    // Loading reads in the exact same order to restore the network state.
    // This allows training once and reusing the model indefinitely.
    void saveWeights(string filename);
    void loadWeights(string filename);
};