#include "Network.h"
#include "Layer.h"
#include <vector>

using std::vector;

Network::Network(vector<int> sizes){
    for ( int i=1 ; i < sizes.size() ; i++){
        Layer l( sizes[i] , sizes[i-1] );
        l.get_weights().randomize(-1.0,1.0);
        layers.push_back(l);
    }
}

Matrix Network::forward( const Matrix& input ){
    Matrix current = input;
    for ( int i=0 ; i < layers.size()-1 ; i++){
    current = layers[i].activate_hidden_layers(layers[i].forward(current));
    }
    current = layers.back().activate_last_layer(layers.back().forward(current));
    return current;
}