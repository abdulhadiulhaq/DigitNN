#pragma once
#include "Matrix.h"
#include "Layer.h"
#include <iostream>
#include <vector>

using std::vector;

class Network{
    public:
    Matrix input;
    vector<Layer> layers;
    vector<Matrix> Z;
    vector<Matrix> A;
    Network(vector<int> sizes);
    Matrix elementwise_multiply( const Matrix& a,  const Matrix& b );
    Matrix forward( const Matrix& input);
    void train(const Matrix& input, const Matrix& expected, double learningRate);
    void saveWeights(string filename);
    void loadWeights(string filename);
};