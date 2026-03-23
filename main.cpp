#include "Network.h"
#include <iostream>
using namespace std;

int main() {
    Network net({784, 16, 16, 10});
    
    Matrix input(784, 1);
    input.randomize(0.0, 1.0);
    
    Matrix output = net.forward(input);
    cout << "Network output (10x1):\n" << output << "\n";
    
    return 0;
}