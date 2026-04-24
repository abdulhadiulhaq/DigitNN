#include "Network.h"
#include "Layer.h"
#include "Matrix.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

// CONSTRUCTOR
// Builds the network layer by layer from the sizes vector.
// For sizes = {784, 128, 64, 10}:
//   i=1: Layer(128, 784)  → 128 neurons receiving 784 inputs
//   i=2: Layer(64,  128)  → 64 neurons receiving 128 inputs
//   i=3: Layer(10,  64)   → 10 neurons receiving 64 inputs
//
// After creating each layer, weights are randomized to small values (-0.1, 0.1).
// Small initial weights are important:
//   - Prevents sigmoid saturation (outputs stuck near 0 or 1)
//   - Keeps gradients from vanishing or exploding at the start
// Biases start at 0 (default from Matrix constructor) which is fine.
Network::Network(vector<int> sizes){
    for ( int i=1 ; i < sizes.size() ; i++){
        Layer l( sizes[i] , sizes[i-1] );
        l.get_weights().randomize(-0.1,0.1);
        layers.push_back(l);
    }
}

// ELEMENTWISE_MULTIPLY
// Multiplies two column vectors (n×1) element by element.
// Result[i] = a[i] * b[i]
//
// This is the Hadamard product, used in backprop to combine two derivatives:
//   delta = propagated_error ⊙ activation_derivative
//
// For the output layer:
//   delta = (A[last] - expected) ⊙ (A[last] * (1 - A[last]))
//
// For hidden layers:
//   delta = (W_next^T * delta_next) ⊙ relu_derivative(Z[i])
//
// Note: Only handles column vectors (single column). For full generality
// you'd loop over columns too, but all our activations are (n×1).
NN::Matrix Network::elementwise_multiply( const NN::Matrix& a,  const NN::Matrix& b ){
    NN::Matrix temp( a.get_row() , a.get_column());
    for( int i = 0 ; i < a.get_row() ; i++){
        temp.set_element( i , 0 , a.get_matrix_element( i , 0 ) * b.get_matrix_element( i , 0) );
    }
    return temp;
}

// FORWARD PASS
// Propagates input through every layer sequentially, storing intermediate
// values that backprop will need later.
//
// For each hidden layer i (all except last):
//   Z[i] = layers[i].forward(current)      → linear: W*x + b
//   A[i] = activate_hidden_layers(Z[i])    → ReLU
//   current = A[i]                         → feed into next layer
//
// For the final layer:
//   Z[last] = layers.back().forward(current)
//   A[last] = activate_last_layer(Z[last]) → Sigmoid (outputs 0-1)
//
// Returns A[last] — the network's output: 10 values representing confidence
// for each digit 0-9.
//
// Z and A are cleared at the start of each call so stale values from the
// previous sample don't interfere.
NN::Matrix Network::forward(const NN::Matrix& input) {
    Z.clear();
    A.clear();
    NN::Matrix current = input;
    for (int i = 0; i < layers.size() - 1; i++) {
        NN::Matrix z = layers[i].forward(current);
        NN::Matrix a = layers[i].activate_hidden_layers(z);
        Z.push_back(z);
        A.push_back(a);
        current = a;
    }
    NN::Matrix z = layers.back().forward(current);
    NN::Matrix a = layers.back().activate_last_layer(z);
    Z.push_back(z);
    A.push_back(a);
    return a;
}

// TRAIN — Full backpropagation and weight update
//
// Mathematical foundation: Gradient Descent with chain rule.
// We want to minimize loss L = (1/n) * Σ(expected - actual)²
// by adjusting weights and biases in the direction that reduces L.
//
// STEP 1: Forward pass
//   Calls forward(input) to fill Z and A for all layers.
//   We need A[last] for the output error and A[i] for weight gradients.
//
// STEP 2: Output layer delta
//   delta = dL/dZ[last] = (A[last] - expected) * sigmoid'(Z[last])
//   sigmoid'(Z) = A * (1 - A)   [we use A[last] since A = sigmoid(Z)]
//   Combined:  delta = (A[last] - expected) ⊙ (A[last] * (1 - A[last]))
//   Shape: (10×1)
//
// STEP 3: Update last layer weights and biases
//   dL/dW = delta * A[last-1]^T    (outer product)
//   W = W - lr * dL/dW
//   B = B - lr * delta
//   prev_A is A[last-1] if it exists, else the raw input (for single-layer nets)
//
// STEP 4: Loop backwards through hidden layers
//   For layer i from (last-1) down to 0:
//     Propagate delta: propagated = W[i+1]^T * delta
//       (W[i+1]^T is the transpose of the NEXT layer's weights)
//     Compute ReLU derivative: relu_deriv[j] = Z[i][j] > 0 ? 1 : 0
//     New delta: delta = propagated ⊙ relu_deriv
//     Update: Wi -= lr * delta * prev^T
//             Bi -= lr * delta
//     prev is A[i-1] for i>0, or the raw input for i=0
void Network::train(const NN::Matrix& input, const NN::Matrix& expected, double learningRate){

    // Step 1 — forward pass fills Z and A
    NN::Matrix output = forward(input);
    int last = A.size()-1;

    // Step 2 — compute output delta
    NN::Matrix diff = A[last] - expected;  // error: predicted - target
 
    // sigmoid derivative: A * (1 - A), computed element-wise
    NN::Matrix sig_deriv( A[last].get_row() , 1 );
    for( int i = 0 ; i < A[last].get_row() ; i++ ){
        double a = A[last].get_matrix_element( i , 0 );
        sig_deriv.set_element( i , 0 , a * (1 - a) );
    }

    // delta for output layer: chain rule combines error and sigmoid derivative
    NN::Matrix delta = elementwise_multiply( diff , sig_deriv );

    // Step 3 — update last layer
    // prev_A is what fed INTO the last layer
    NN::Matrix previous_A = ( last >= 1) ? A[last-1] : input;

    NN::Matrix& W = layers[last].get_weights();
    NN::Matrix& B = layers[last].get_biases();

    // Gradient of weights = outer product of delta and prev_A
    // delta: (10×1), prev_A^T: (1×64) → product: (10×64) matches W shape
    W = W - ( delta * previous_A.transpose() ) * learningRate;
    B = B - delta * learningRate;

    // Step 4 — backpropagate through hidden layers (right to left)
    for ( int i = last - 1 ; i >= 0 ; i-- ){
        // Propagate delta backwards through the next layer's weights
        // W_next^T: (prev_size × current_size), delta: (current_size × 1)
        // Result: (prev_size × 1) — error signal for this layer
        NN::Matrix W_next = layers[ i + 1 ].get_weights();
        NN::Matrix propagated = W_next.transpose() * delta;

        // ReLU derivative: 1 where Z was positive (neuron was active), 0 elsewhere
        NN::Matrix relu_deriv( Z[i].get_row() , Z[i].get_column() );
        for ( int j = 0 ; j < Z[i].get_row() ; j++ ){
            double val = Z[i].get_matrix_element( j , 0 ) > 0 ? 1.0 : 0.0;
            relu_deriv.set_element( j , 0 , val );
        }
        // New delta for this layer: combine propagated error with local derivative
        delta = elementwise_multiply( propagated , relu_deriv );

        // Input to this layer: A[i-1] for hidden layers, raw input for first layer
        NN::Matrix prev = ( i > 0 ) ? A[ i - 1 ] : input;
        NN::Matrix& Wi = layers[i].get_weights();
        NN::Matrix& Bi = layers[i].get_biases();

        Wi = Wi - ( delta * prev.transpose() ) * learningRate;
        Bi = Bi - delta * learningRate;
    }

}

// SAVEWEIGHTS
// Writes all weights and biases to a plain text file, one double per line.
// Iterates layers in order, writing weights row-by-row then biases.
// The exact same iteration order is used in loadWeights to restore correctly.
//
// File size for {784, 128, 64, 10}:
//   Layer 0: 128*784 + 128 = 100,480 values
//   Layer 1:  64*128 +  64 =   8,256 values
//   Layer 2:  10*64  +  10 =     650 values
//   Total: ~109,386 lines ≈ 1-2 MB
void Network::saveWeights(string filename) {
    ofstream file(filename);
    for (int i = 0; i < layers.size(); i++) {
        NN::Matrix& W = layers[i].get_weights();
        NN::Matrix& B = layers[i].get_biases();

        // Write all weight values
        for (int r = 0; r < W.get_row(); r++)
            for (int c = 0; c < W.get_column(); c++)
                file << W.get_matrix_element(r, c) << "\n";
        
        // Write all biases values
        for (int r = 0; r < B.get_row(); r++)
            file << B.get_matrix_element(r, 0) << "\n";
    }
}

// LOADWEIGHTS
// Reads weights and biases from file in the same order saveWeights wrote them.
// The network must be constructed with the SAME architecture ({784, 128, 64, 10})
// as when the weights were saved — otherwise dimensions won't match and values
// will be loaded into wrong layers, causing crashes or garbage predictions.
void Network::loadWeights(string filename) {
    ifstream file(filename);
    for (int i = 0; i < layers.size(); i++) {
        NN::Matrix& W = layers[i].get_weights();
        NN::Matrix& B = layers[i].get_biases();
        for (int r = 0; r < W.get_row(); r++)
            for (int c = 0; c < W.get_column(); c++) {
                double val; file >> val;
                W.set_element(r, c, val);
            }
        for (int r = 0; r < B.get_row(); r++) {
            double val; file >> val;
            B.set_element(r, 0, val);
        }
    }
}