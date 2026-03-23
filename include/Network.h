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
    Network(vector<int> sizes);
    Matrix forward( const Matrix& input);
};