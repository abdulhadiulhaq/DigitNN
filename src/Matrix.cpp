#include "Matrix.h"
#include <random>


namespace NN {

    // CONSTRUCTOR
    // Initializes number_rows and number_columns, then builds the 2D data
    // vector filled with 0.0.
    // data(number_rows, vector<double>(number_columns, 0.0)) creates:
    //   - number_rows outer vectors
    //   - each containing number_columns doubles, all set to 0.0
    // This means a freshly created Matrix is always zero-initialized —
    // no garbage values, no undefined behavior.

    Matrix::Matrix(int rows,int columns) : number_rows(rows),number_columns(columns), data(number_rows,vector<double>(number_columns,0.0)){}
    
    // SETTERS
    // These only update the integer fields — they do NOT resize 'data'.
    // They are intentionally low-level. Only operator= calls them alongside
    // a proper data.assign() resize. Do not call them in isolation unless
    // you also handle the data vector manually.
    void Matrix::set_row(int r) { this->number_rows=r;}
    void Matrix::set_column(int c){ this->number_columns=c;}

    // Sets a single element. No bounds checking — caller is responsible.
    // data[r][c] = d directly writes into the 2D vector.
    void Matrix::set_element(int r, int c, double d){ this->data[r][c]=d;}
    
    // GETTERS — trivial accessors, marked const for use in const contexts
    int Matrix::get_row() const { return this->number_rows;}
    int Matrix::get_column() const { return this->number_columns;}
    double Matrix::get_matrix_element(int row,int column) const {
        return data[row][column];
    }

    // OPERATOR* — Matrix Multiplication
    // Standard O(n³) matrix multiply. For matrices A (m×k) and B (k×n):
    //   C[i][k] += A[i][j] * B[j][k]  for all j
    //
    // Loop order is i→j→k (not i→k→j) which is cache-friendly for row-major
    // storage because we iterate along B's columns in the innermost loop.
    //
    // Usage in this project:
    //   Layer::forward()  → weights(n×m) * input(m×1)  = output(n×1)
    //   Backprop          → W^T(m×n)    * delta(n×1)   = propagated(m×1)
    //   Weight gradient   → delta(n×1)  * prev^T(1×m)  = grad(n×m)
    //
    // Returns Matrix(0,0) if dimensions are incompatible instead of throwing.
    // This is a silent failure — watch for 0x0 matrices in debug output.
    Matrix Matrix::operator*(const Matrix& other) const {
        if(this->number_columns!=other.get_row()){
            return Matrix(0,0);
        }
        Matrix temp(this->number_rows,other.get_column());
        // A[i][j]*B[j][k]=C[i][k]
        for (int i=0;i<this->number_rows;i++){

            for(int j=0;j<this->number_columns;j++){

                for(int k=0;k<other.get_column();k++){
                    temp.set_element( i , k , this->get_matrix_element( i , j )*other.get_matrix_element( j , k ) + temp.get_matrix_element( i , k ) );
                }
            }
        }
        return temp;
    }

    // OPERATOR* — Scalar Multiplication
    // Multiplies every element by a scalar. Used to scale gradients by the
    // learning rate during weight updates:
    //   W = W - (delta * prev^T) * learningRate
    // Simple O(m*n) loop over all elements.
    Matrix Matrix::operator*(double scalar) const {
        Matrix temp(number_rows, number_columns);
        for (int i = 0; i < number_rows; i++)
            for (int j = 0; j < number_columns; j++)
                temp.set_element(i, j, data[i][j] * scalar);
        return temp;
    }

    // OPERATOR+ — Element-wise Addition
    // Adds two matrices of identical dimensions element by element.
    // Used in Layer::forward() to add the bias vector to the weighted sum:
    //   Z = weights * input + biases
    // biases is a (n×1) column vector, input result is also (n×1) — they match.
    // Throws a const char* string if dimensions differ (caught in training loop).
    Matrix Matrix::operator+(const Matrix& other) const {
        if(this->get_row()!=other.get_row() || this->get_column()!=other.get_column()){
            throw "[!] Error! Order does not match!\n";
        }
        Matrix temp(this->get_row(),this->get_column());
        for(int i=0 ; i<this->get_row() ; i++){

            for(int j=0 ; j<this->get_column() ; j++){
                temp.set_element( i , j , this->get_matrix_element( i , j )+other.get_matrix_element( i , j ));

            }
        }
        return temp;
    }

    // OPERATOR- — Element-wise Subtraction
    // Subtracts other from this element by element. Same dimension requirement.
    // Used in backprop to compute the output error:
    //   diff = A[last] - expected   (predicted output minus one-hot target)
    // Also used for weight updates:
    //   W = W - gradient
    Matrix Matrix::operator-(const Matrix& other) const {
        if(this->get_row()!=other.get_row() || this->get_column()!=other.get_column()){
            throw "[!] Error! Order does not match!\n";
        }
        Matrix temp(this->get_row(),this->get_column());
        for(int i=0 ; i<this->get_row() ; i++){

            for(int j=0 ; j<this->get_column() ; j++){
                temp.set_element( i , j , this->get_matrix_element( i , j )-other.get_matrix_element( i , j ));

            }
        }
        return temp;
    }

    // TRANSPOSE
    // Creates a new matrix with rows and columns swapped.
    // If this is (m×n), result is (n×m), where result[i][j] = this[j][i].
    //
    // Why we need it in backprop:
    //   1. Weight gradient shape must match W shape:
    //      delta is (n×1), prev_A is (m×1)
    //      delta * prev_A^T = (n×1)*(1×m) = (n×m) ✓ matches W shape
    //
    //   2. Propagating delta to previous layer:
    //      W is (n×m), delta is (n×1)
    //      W^T * delta = (m×n)*(n×1) = (m×1) ✓ matches previous layer size
    Matrix Matrix::transpose() const{
        Matrix temp(this->get_column(),this->get_row());
        for( int i = 0 ; i < this->get_column() ; i++){

            for( int j = 0 ; j < this->get_row() ; j++){
                temp.set_element( i , j ,this->get_matrix_element( j , i ));
            }
        }
        return temp;
    }

    // OPERATOR<< — Pretty Print
    // Prints every row on a new line with spaces between elements.
    // Friend function so it can access private 'data' directly.
    // Usage: cout << myMatrix;
    ostream& operator<<(ostream& os, const Matrix& m){
        for ( int i=0 ; i < m.get_row() ; i++){

            for ( int j=0 ; j < m.get_column() ; j++){
                os << " " << m.get_matrix_element( i , j ) << " ";
            }
            os << "\n";
        }
        return os;
    }

    // OPERATOR= — Deep Copy Assignment
    // Copies all data from 'other' into 'this'.
    //
    // CRITICAL: We call data.assign() to RESIZE the data vector before copying.
    // Without this, if 'this' was smaller than 'other', accessing data[i][j]
    // for the new larger indices would crash with an assertion failure:
    //   "Assertion '__n < this->size()' failed"
    // This was a real bug in earlier versions of this project.
    //
    // Self-assignment guard (this == &other) prevents clearing our own data
    // before copying it, which would result in a zeroed matrix.
    Matrix& Matrix::operator=(const Matrix& other){
        if(this == &other) return *this;
        number_rows = other.get_row();
        number_columns = other.get_column();
        data.assign(number_rows, vector<double>(number_columns, 0.0));
        for (int i = 0; i < number_rows; i++)
            for (int j = 0; j < number_columns; j++)
                data[i][j] = other.get_matrix_element(i, j);
        
        return *this;
    }

    // RANDOMIZE
    // Fills all elements with uniform random doubles in [min, max].
    //
    // Why randomize weights at all?
    //   If all weights start at 0, every neuron computes the same output and
    //   receives the same gradient — the network can never learn distinct
    //   features. Random initialization breaks this symmetry.
    //
    // Why small values like (-0.1, 0.1)?
    //   Large initial weights push activations into the flat regions of sigmoid
    //   (near 0 or 1), where gradients are near zero — the vanishing gradient
    //   problem. Small weights keep activations in the sensitive middle range.
    //
    // Uses mt19937 (Mersenne Twister) seeded by random_device for
    // non-deterministic initialization each run.
    void Matrix::randomize(double min, double max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min, max);
        for (int i = 0; i < number_rows; i++)
            for (int j = 0; j < number_columns; j++)
                data[i][j] = dist(gen);
    }
}