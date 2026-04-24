#include "Network.h"
#include "DataLoader.h"
#include <iostream>
#include <ctime>
using namespace std;

NN::Matrix oneHot(int label) {
    NN::Matrix m(10, 1);
    m.set_element(label, 0, 1.0);
    return m;
}

double loss(const NN::Matrix& expected, const NN::Matrix& actual) {
    double total = 0.0;
    for (int i = 0; i < expected.get_row(); i++) {
        double val = expected.get_matrix_element(i, 0) - actual.get_matrix_element(i, 0);
        total += val * val;
    }
    return total / expected.get_row();
}

int main() {
    int epochs = 50;
    Network net({784, 128, 64, 10});
    double lr = 0.0001;

    vector<NN::Matrix> images = loadImages("../data/train-images.idx3-ubyte");
    vector<int> labels = loadLabels("../data/train-labels.idx1-ubyte");

    for (int e = 0; e < epochs; e++) {
        double totalLoss = 0.0;
        time_t start = time(0);
        for (int i = 0; i < 60000; i++) {
            NN::Matrix expected = oneHot(labels[i]);
            NN::Matrix output = net.forward(images[i]);
            totalLoss += loss(expected, output);
            net.train(images[i], expected, lr);
            if (i % 1000 == 0 && i > 0) {
                cout << "Epoch " << e+1 << " Sample " << i << " Loss: " << totalLoss/i << "\n";
            }
        }
        cout << "Epoch " << e+1 << " complete. Avg Loss: " << totalLoss/60000
             << " Time: " << time(0)-start << "s\n";
    }

    net.saveWeights("../data/weights.txt");
    cout << "Weights saved.\n";
    return 0;
}