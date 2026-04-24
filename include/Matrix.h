#pragma once
#include <vector>
#include <iostream>

using std::vector;
using std::ostream;


// We wrap our Matrix inside the NN namespace to prevent name collision with
// Raylib's own 'Matrix' typedef (a 4x4 float matrix used for 3D transforms).
// Any file that uses our Matrix must either write NN::Matrix or place
// "using namespace NN;" at the top (avoid in headers to prevent pollution).

namespace NN{
    class Matrix {
    private:
        int number_rows;     // how many rows this matrix has
        int number_columns;      // how many columns this matrix has

        // The actual data stored as a 2D vector of doubles.
        // data[row][col] accesses the element at a given position.
        // Initialized to all 0.0 in the constructor.
        vector<vector<double>> data;
    public:
        
        // CONSTRUCTOR
        // Creates a matrix of given size, filled with 0.0.
        // Default arguments (rows=0, columns=0) allow declaring an empty Matrix
        // without arguments, which is needed by Layer to declare weight/bias
        // members before they are assigned.
        // Example: Matrix m(784, 1) creates a 784-row, 1-column column vector.

        Matrix(int rows=0,int columns=0);
        
        // SETTERS
        // Directly overwrite the stored row/column count.
        // Used internally by operator= to resize a matrix after assignment.
        // Caution: calling set_row/set_column without resizing 'data' will
        // cause out-of-bounds access — always resize 'data' alongside these.

        void set_row(int r);
        void set_column(int c);
        
        // Sets the value at position (r, c) to d.
        // Used everywhere: building input vectors, updating weights, etc.
        void set_element(int r, int c, double d);
        
        // GETTERS
        // Return the dimensions or a specific element value.
        // Marked const so they can be called on const Matrix references
        // (e.g., inside operator* which takes const Matrix& other).
        int get_row() const;
        int get_column() const;
        double get_matrix_element(int row,int column) const;

        // OPERATOR* — Matrix Multiplication
        // Computes standard matrix product: C = A * B
        // Requirement: A.columns must equal B.rows
        // Result dimensions: (A.rows) x (B.columns)
        // Used in Layer::forward() as: output = weights * input
        // Also used in backprop as: grad = W^T * delta
        // Returns Matrix(0,0) silently if dimensions don't match — be careful.

        Matrix operator*(const Matrix& other) const;

        // OPERATOR* — Scalar Multiplication
        // Multiplies every element by a scalar double.
        // Used in backprop to scale gradients by the learning rate:
        //   W = W - (delta * prev^T) * learningRate

        Matrix operator*(double scalar) const;

        // OPERATOR+ — Matrix Addition
        // Adds two matrices element-by-element. Dimensions must match exactly.
        // Used in Layer::forward() to add biases: Z = W*input + biases
        // Throws a string error if dimensions don't match.

        Matrix operator+(const Matrix& other) const;

        // OPERATOR- — Matrix Subtraction
        // Subtracts element-by-element. Dimensions must match exactly.
        // Used in backprop to compute the error: diff = A[last] - expected
        // Also used in weight update: W = W - gradient
        // Throws a string error if dimensions don't match.
        Matrix operator-(const Matrix& other) const;

        // TRANSPOSE
        // Flips the matrix along its diagonal: rows become columns and vice versa.
        // A matrix of size (m x n) becomes (n x m).
        // Critical in backprop:
        //   - Weight gradient: delta * prev_A^T  →  makes dimensions match for W update
        //   - Delta propagation: W^T * delta     →  passes error to previous layer
        Matrix transpose() const;

        // OPERATOR<< — Stream Output
        // Prints the matrix to an output stream (e.g., cout) row by row.
        // Declared as a friend so it can access private 'data' directly.
        // Useful for debugging: cout << myMatrix;

        friend ostream& operator<<(ostream& os, const Matrix& m);

        // OPERATOR= — Assignment
        // Deep copies all data from 'other' into this matrix.
        // IMPORTANT: Also resizes 'data' using data.assign() before copying.
        // Without the resize, assigning a larger matrix into a smaller one
        // would cause out-of-bounds access on the data vector — this was a
        // critical bug that caused crashes during backprop weight updates.

        Matrix& operator=(const Matrix& other);

        // RANDOMIZE
        // Fills every element with a random double in [min, max].
        // Uses C++ <random> with mt19937 (Mersenne Twister) for quality randomness.
        // Called during Network construction to initialize all layer weights.
        // Small initial weights (e.g., -0.1 to 0.1) prevent vanishing/exploding
        // gradients at the start of training.
        
        void randomize(double min, double max);
    };
}