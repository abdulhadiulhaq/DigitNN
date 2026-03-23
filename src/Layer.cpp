#include "Layer.h"
#include "Matrix.h"
#include <cmath>

using std::fmax;

void Layer::set_biases(const Matrix& m){ biases = m;}

void Layer::set_weights(const Matrix& m){ weights = m;}

void Layer::set_current_neurons( int nc ){ current_neurons = nc;}

void Layer::set_previous_neurons( int np ){ previous_neurons = np;}

Matrix& Layer::get_biases(){ return biases;}

Matrix& Layer::get_weights(){ return weights;}

int Layer::get_current_neurons(){ return current_neurons;}

int Layer::get_previous_neurons(){ return previous_neurons;}

Layer::Layer(int nc=0, int np=0) : current_neurons(nc) , previous_neurons(np) , weights(current_neurons, previous_neurons) , biases(current_neurons,1){}

Matrix Layer::activate_hidden_layers(const Matrix& m) const{
    Matrix temp(m.get_row(),m.get_column());
    for (int i=0; i < m.get_row() ; i++){

        for (int j=0 ; j < m.get_column() ; j++){
            temp.set_element( i , j ,fmax(0,m.get_matrix_element( i , j )));
        }
    }
    return temp;
}
Matrix Layer::forward(const Matrix& input) const{
    return weights * input + biases;
}

Matrix Layer::activate_last_layer(const Matrix& m) const{
    Matrix temp(m.get_row(),m.get_column());
    for (int i=0; i < m.get_row() ; i++){

        for (int j=0 ; j < m.get_column() ; j++){
            temp.set_element( i , j , 1.0/(1.0+exp(-m.get_matrix_element( i , j ))));
        }
    }
    return temp;
}