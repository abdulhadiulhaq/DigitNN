#include "Matrix.h"

Matrix::Matrix(int rows,int columns) : number_rows(rows),number_columns(columns), data(number_rows,vector<double>(number_columns,0.0)){}
    
void Matrix::set_row(int r) { this->number_rows=r;}
    
void Matrix::set_column(int c){ this->number_columns=c;}
    
void Matrix::set_element(int r, int c, double d){ this->data[r][c]=d;}
    
int Matrix::get_row() const { return this->number_rows;}
    
int Matrix::get_column() const { return this->number_columns;}
    
double Matrix::get_matrix_element(int row,int column) const {
    return data[row][column];
}

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

Matrix Matrix::transpose() const{
    Matrix temp(this->get_column(),this->get_row());
    for( int i = 0 ; i < this->get_column() ; i++){

        for( int j = 0 ; j < this->get_row() ; j++){
            temp.set_element( i , j ,this->get_matrix_element( j , i ));
        }
    }
    return temp;
}

ostream& operator<<(ostream& os, const Matrix& m){
    for ( int i=0 ; i < m.get_row() ; i++){

        for ( int j=0 ; j < m.get_column() ; j++){
            os << " " << m.get_matrix_element( i , j ) << " ";
        }
        os << "\n";
    }
    return os;
}

Matrix& Matrix::operator=(const Matrix& other){
    if( this == &other ){
        return *this;
    }
    else{
        this->set_row(other.get_row());
        this->set_column(other.get_column());
        for (int i = 0 ; i < this->get_row() ; i++){

            for (int j = 0 ; j < this->get_column() ; j++){
                this->set_element( i , j , other.get_matrix_element( i , j ));
            }
        }
    }
    return *this;
}