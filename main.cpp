#include "Matrix.h"
#include <iostream>
using namespace std;

int main() {
    Matrix a(2, 3);
    a.set_element(0, 0, 1); a.set_element(0, 1, 2); a.set_element(0, 2, 3);
    a.set_element(1, 0, 4); a.set_element(1, 1, 5); a.set_element(1, 2, 6);

    Matrix b(3, 2);
    b.set_element(0, 0, 7); b.set_element(0, 1, 8);
    b.set_element(1, 0, 9); b.set_element(1, 1, 10);
    b.set_element(2, 0, 11); b.set_element(2, 1, 12);

    cout << "A:\n" << a << "\n";
    cout << "B:\n" << b << "\n";
    cout << "A*B:\n" << a * b << "\n";
    cout << "Transpose of A:\n" << a.transpose() << "\n";
}