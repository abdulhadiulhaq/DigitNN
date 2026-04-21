#include "Network.h"
#include "DataLoader.h"
#include <iostream>
using namespace std;

Matrix oneHot(int label) {
    Matrix m(10, 1);
    m.set_element(label, 0, 1.0);
    return m;
}
double loss(const Matrix& expected, const Matrix& actual) {
    // MSE = (1/n) * sum of (expected[i] - actual[i])²
    double total = 0.0;
    Matrix diff = expected - actual;
    for ( int i=0 ; i < expected.get_row() ; i++){
        double val = diff.get_matrix_element( i , 0 );
        total += val * val;
    }
    return total/diff.get_row();
}
int main() {
    int epochs = 10;
    Network net({ 784 , 128 , 16 , 10 });
    double lr = 0.01;
    vector<Matrix> images = loadImages("../data/train-images.idx3-ubyte");
    vector<int> labels = loadLabels("../data/train-labels.idx1-ubyte");
    for ( int e = 0 ; e < epochs ; e++ ){
    double totalLoss= 0.0;
        for ( int i = 0 ; i < 60000 ; i++ ){
            Matrix expected = oneHot(labels[i]);
            Matrix output = net.forward(images[i]);
            net.train(images[i], expected, lr);
            totalLoss += loss(expected, output);
            if (i % 1000 == 0 && i > 0){
                cout << "Epoch " << e+1 << " Sample " << i << " Loss: " << totalLoss/i << "\n";
            }
        }
        cout << "Epoch " << e+1 << " complete. Avg Loss: " << totalLoss/60000 << "\n";
    }
    net.saveWeights("../data/weights.txt");
    cout << "Weights saved.\n";
    return 0;
}