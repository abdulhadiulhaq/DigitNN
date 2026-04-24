#pragma once
#include "Matrix.h"
#include "Layer.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Network{
    public:
    NN::Matrix input;
    vector<Layer> layers;
    vector<NN::Matrix> Z;
    vector<NN::Matrix> A;
    Network(vector<int> sizes);
    NN::Matrix elementwise_multiply( const NN::Matrix& a,  const NN::Matrix& b );
    NN::Matrix forward( const NN::Matrix& input);
    void train(const NN::Matrix& input, const NN::Matrix& expected, double learningRate);
    void saveWeights(string filename);
    void loadWeights(string filename);
};