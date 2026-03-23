#pragma once
#include "Matrix.h"

class Layer{
    int current_neurons;
    int previous_neurons;
    Matrix weights;
    Matrix biases;
    public:
        Layer(int nc, int np);
        Matrix forward(const Matrix& input) const;
        Matrix activate_hidden_layers(const Matrix& m) const;
        Matrix activate_last_layer(const Matrix& m) const;
        Matrix& get_weights();
        Matrix& get_biases();
        int get_current_neurons();
        int get_previous_neurons();
        void set_weights(const Matrix& m);
        void set_biases(const Matrix& m);
        void set_current_neurons(int nc);
        void set_previous_neurons(int np);
};