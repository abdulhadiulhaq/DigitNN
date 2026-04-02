#include "Network.h"
#include "DataLoader.h"
#include <iostream>
using namespace std;

int main() {
    vector<Matrix> images = loadImages("../data/train-images.idx3-ubyte");
    vector<int> labels = loadLabels("../data/train-labels.idx1-ubyte");
    cout << "First label: " << labels[0] << "\n";
    cout << "First 10 pixels of first image:\n";
    for (int i = 0; i < 10; i++) {
        cout << images[0].get_matrix_element(i, 0) << "\n";
    }
    return 0;
}