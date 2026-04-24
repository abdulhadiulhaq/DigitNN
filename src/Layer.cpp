#include "Layer.h"
#include "Matrix.h"
#include <cmath>       // for exp() in sigmoid, fmax() in ReLU

using std::fmax;

    // TRIVIAL SETTERS AND GETTERS
    // These are straightforward wrappers. Getters return references (not copies)
    // so the caller can modify weights/biases in-place during training.
    void Layer::set_biases(const NN::Matrix& m){ biases = m;}
    void Layer::set_weights(const NN::Matrix& m){ weights = m;}
    void Layer::set_current_neurons( int nc ){ current_neurons = nc;}
    void Layer::set_previous_neurons( int np ){ previous_neurons = np;}

    NN::Matrix& Layer::get_biases(){ return biases;}
    NN::Matrix& Layer::get_weights(){ return weights;}
    int Layer::get_current_neurons(){ return current_neurons;}
    int Layer::get_previous_neurons(){ return previous_neurons;}

    // CONSTRUCTOR
    // Member initializer list sets all four members in the correct order.
    // weights is initialized as (nc × np): nc neurons, each with np input weights.
    // biases is initialized as (nc × 1): one bias per neuron, all zero.
    //
    // After construction, the caller (Network constructor) calls:
    //   l.get_weights().randomize(-0.1, 0.1)
    // to break weight symmetry before training begins.
    Layer::Layer(int nc=0, int np=0) : current_neurons(nc) , previous_neurons(np) , weights(current_neurons, previous_neurons) , biases(current_neurons,1){}

    // ACTIVATE_HIDDEN_LAYERS — ReLU activation
    // Applied to every hidden layer (all layers except the last).
    //
    // ReLU formula: f(x) = max(0, x)
    //   - Negative values become 0 (neuron "off")
    //   - Positive values pass through unchanged (neuron "on")
    //
    // ReLU derivative (used in backprop):
    //   f'(x) = 1 if x > 0
    //   f'(x) = 0 if x <= 0
    // This is why we store Z (pre-activation values) during the forward pass —
    // backprop needs Z to know which neurons were active (Z > 0) to compute
    // the correct gradient.
    //
    // fmax(0, x) is used instead of a conditional for clarity and compiler
    // optimization potential.
    NN::Matrix Layer::activate_hidden_layers(const NN::Matrix& m) const{
        NN::Matrix temp(m.get_row(),m.get_column());
        for (int i=0; i < m.get_row() ; i++){

            for (int j=0 ; j < m.get_column() ; j++){
                temp.set_element( i , j ,fmax(0,m.get_matrix_element( i , j )));
            }
        }
        return temp;
    }

    // FORWARD — Linear transformation (no activation)
    // Computes Z = weights * input + biases
    //
    // input:  (previous_neurons × 1) column vector
    // output: (current_neurons  × 1) column vector
    //
    // This is the raw "score" before activation — stored as Z in the Network's
    // forward pass so backprop can use it for derivative calculations later.
    // Activation is applied separately by activate_hidden_layers() or
    // activate_last_layer() in Network::forward().
    NN::Matrix Layer::forward(const NN::Matrix& input) const{
        return weights * input + biases;
    }

    // ACTIVATE_LAST_LAYER — Sigmoid activation
    // Applied only to the final output layer (10 neurons for digits 0-9).
    //
    // Sigmoid formula: f(x) = 1 / (1 + e^(-x))
    //   - Output is always in range (0, 1)
    //   - Large positive x → output near 1.0 (confident "yes")
    //   - Large negative x → output near 0.0 (confident "no")
    //   - x = 0 → output = 0.5 (uncertain)
    //
    // Sigmoid derivative (used in backprop output delta):
    //   f'(x) = f(x) * (1 - f(x)) = A * (1 - A)
    // This is why we store A[last] (post-activation output) — backprop computes
    // sig_deriv = A[last] * (1 - A[last]) directly from the stored activation.
    NN::Matrix Layer::activate_last_layer(const NN::Matrix& m) const{
        NN::Matrix temp(m.get_row(),m.get_column());
        for (int i=0; i < m.get_row() ; i++){

            for (int j=0 ; j < m.get_column() ; j++){
                temp.set_element( i , j , 1.0/(1.0+exp(-m.get_matrix_element( i , j ))));
            }
        }
        return temp;
    }