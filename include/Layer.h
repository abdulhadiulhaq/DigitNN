#pragma once
#include "Matrix.h"

class Layer{
    int current_neurons;
    int previous_neurons;
    NN::Matrix weights;
    NN::Matrix biases;
    public:
        Layer(int nc, int np);
        NN::Matrix forward(const NN::Matrix& input) const;
        NN::Matrix activate_hidden_layers(const NN::Matrix& m) const;
        NN::Matrix activate_last_layer(const NN::Matrix& m) const;
        NN::Matrix& get_weights();
        NN::Matrix& get_biases();
        int get_current_neurons();
        int get_previous_neurons();
        void set_weights(const NN::Matrix& m);
        void set_biases(const NN::Matrix& m);
        void set_current_neurons(int nc);
        void set_previous_neurons(int np);
};