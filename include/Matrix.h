#pragma once
#include <vector>
#include <iostream>
using std::vector;
using std::ostream;
class Matrix {
private:
    int number_rows;
    int number_columns;
    vector<vector<double>> data;
public:
    Matrix(int rows=0,int columns=0);
    
    void set_row(int r);
    
    void set_column(int c);
    
    void set_element(int r, int c, double d);
    
    int get_row() const;
    
    int get_column() const;
    
    double get_matrix_element(int row,int column) const;

    Matrix operator*(const Matrix& other) const;

    Matrix operator+(const Matrix& other) const;

    Matrix transpose() const;

    friend ostream& operator<<(ostream& os, const Matrix& m);

    Matrix& operator=(const Matrix& other);
};