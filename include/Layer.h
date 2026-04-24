#pragma once
#include "Matrix.h"

// Note: We use NN:: prefix throughout instead of "using namespace NN" to avoid
// polluting the global namespace in headers — especially important since Raylib
// also defines a Matrix type in the global namespace.
 
class Layer{

    // PRIVATE MEMBERS
    // current_neurons:  number of neurons in THIS layer (output size)
    // previous_neurons: number of neurons in the PREVIOUS layer (input size)
    //
    // weights: Matrix of shape (current_neurons × previous_neurons)
    //   Each row i contains the weights connecting all previous neurons to
    //   neuron i in this layer. weights[i][j] = weight from prev neuron j
    //   to current neuron i.
    //
    // biases: Matrix of shape (current_neurons × 1)
    //   One bias per neuron. Added after the weighted sum to shift the
    //   activation function left or right, giving each neuron a threshold.
    int current_neurons;
    int previous_neurons;
    NN::Matrix weights;
    NN::Matrix biases;
    public:

        // CONSTRUCTOR
        // nc = number of neurons in this layer (current)
        // np = number of neurons in the previous layer (input dimensionality)
        // Initializes weights as (nc × np) and biases as (nc × 1), all zeros.
        // Weights are randomized externally after construction via get_weights().randomize()
        Layer(int nc = 0 , int np = 0 );

        // FORWARD PASS — Linear transformation only (no activation)
        // Computes: Z = weights * input + biases
        // input shape: (previous_neurons × 1)
        // output shape: (current_neurons × 1)
        //
        // Activation is deliberately SEPARATE from this method so that both Z
        // (pre-activation) and A (post-activation) can be stored during the forward
        // pass. Backpropagation needs both values — Z for computing derivatives,
        // A for computing weight gradients.
        NN::Matrix forward(const NN::Matrix& input) const;

        // ACTIVATION — ReLU (Rectified Linear Unit) for hidden layers
        // ReLU(x) = max(0, x)
        // Applied element-wise to every value in the matrix.
        //
        // Why ReLU for hidden layers?
        //   - Simple and fast to compute
        //   - Derivative is either 0 or 1 — no vanishing gradient in positive range
        //   - Sparse activation: ~50% of neurons output 0, making the network efficient
        //   - Works well in practice for most classification tasks
        NN::Matrix activate_hidden_layers(const NN::Matrix& m) const;

        // ACTIVATION — Sigmoid for the output layer
        // Sigmoid(x) = 1 / (1 + e^(-x))
        // Squashes output to range (0, 1) for each of the 10 output neurons.
        //
        // Why Sigmoid for the output?
        //   - Outputs are interpretable as probabilities (0 to 1)
        //   - Combined with MSE loss, its derivative A*(1-A) integrates cleanly
        //     into the backprop delta calculation
        //
        // Note: Softmax + cross-entropy is theoretically superior for multi-class
        // classification, but Sigmoid + MSE is simpler to implement and still works.
        NN::Matrix activate_last_layer(const NN::Matrix& m) const;

        // Getters — return references so Network::train() can modify weights/biases
        // directly without copying (important for performance and correctness)
        NN::Matrix& get_weights();
        NN::Matrix& get_biases();
        int get_current_neurons();
        int get_previous_neurons();

        // Setters — used by loadWeights() to restore saved state
        void set_weights(const NN::Matrix& m);
        void set_biases(const NN::Matrix& m);
        void set_current_neurons(int nc);
        void set_previous_neurons(int np);
};