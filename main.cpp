#include "Matrix.h"
#include "Layer.h"
#include <iostream>
using namespace std;

int main() {
    // Create a layer: 3 neurons, takes 2 inputs
    Layer l(3, 2);

    // Create input matrix (2x1)
    Matrix input(2, 1);
    input.set_element(0, 0, 0.5);
    input.set_element(1, 0, -0.3);

    // Forward pass
    Matrix z = l.forward(input);
    cout << "Raw output (z):\n" << z << "\n";

    // Apply activations
    Matrix a_hidden = l.activate_hidden_layers(z);
    cout << "After ReLU:\n" << a_hidden << "\n";

    Matrix a_output = l.activate_last_layer(z);
    cout << "After Sigmoid:\n" << a_output << "\n";
}