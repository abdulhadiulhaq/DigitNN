#include "Network.h"
#include "Layer.h"
#include "Matrix.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

Network::Network(vector<int> sizes){
    for ( int i=1 ; i < sizes.size() ; i++){
        Layer l( sizes[i] , sizes[i-1] );
        l.get_weights().randomize(-0.1,0.1);
        layers.push_back(l);
    }
}

NN::Matrix Network::elementwise_multiply( const NN::Matrix& a,  const NN::Matrix& b ){
    NN::Matrix temp( a.get_row() , a.get_column());
    for( int i = 0 ; i < a.get_row() ; i++){
        temp.set_element( i , 0 , a.get_matrix_element( i , 0 ) * b.get_matrix_element( i , 0) );
    }
    return temp;
}

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

void Network::train(const NN::Matrix& input, const NN::Matrix& expected, double learningRate){
    NN::Matrix output = forward(input);
    int last = A.size()-1;

    NN::Matrix diff = A[last] - expected;

    NN::Matrix sig_deriv( A[last].get_row() , 1 );
    for( int i = 0 ; i < A[last].get_row() ; i++ ){
        double a = A[last].get_matrix_element( i , 0 );
        sig_deriv.set_element( i , 0 , a * (1 - a) );
    }
    NN::Matrix delta = elementwise_multiply( diff , sig_deriv );

    NN::Matrix previous_A = ( last >= 1) ? A[last-1] : input;

    NN::Matrix& W = layers[last].get_weights();
    NN::Matrix& B = layers[last].get_biases();

    W = W - ( delta * previous_A.transpose() ) * learningRate;
    B = B - delta * learningRate;

    for ( int i = last - 1 ; i >= 0 ; i-- ){
        NN::Matrix W_next = layers[ i + 1 ].get_weights();
        NN::Matrix propagated = W_next.transpose() * delta;

        NN::Matrix relu_deriv( Z[i].get_row() , Z[i].get_column() );
        for ( int j = 0 ; j < Z[i].get_row() ; j++ ){
            double val = Z[i].get_matrix_element( j , 0 ) > 0 ? 1.0 : 0.0;
            relu_deriv.set_element( j , 0 , val );
        }
        delta = elementwise_multiply( propagated , relu_deriv );

        NN::Matrix prev = ( i > 0 ) ? A[ i - 1 ] : input;
        NN::Matrix& Wi = layers[i].get_weights();
        NN::Matrix& Bi = layers[i].get_biases();

        Wi = Wi - ( delta * prev.transpose() ) * learningRate;
        Bi = Bi - delta * learningRate;
    }

}

void Network::saveWeights(string filename) {
    ofstream file(filename);
    for (int i = 0; i < layers.size(); i++) {
        NN::Matrix& W = layers[i].get_weights();
        NN::Matrix& B = layers[i].get_biases();
        for (int r = 0; r < W.get_row(); r++)
            for (int c = 0; c < W.get_column(); c++)
                file << W.get_matrix_element(r, c) << "\n";
        for (int r = 0; r < B.get_row(); r++)
            file << B.get_matrix_element(r, 0) << "\n";
    }
}

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