#include "DataLoader.h"
#include <fstream>
#include "Matrix.h"
using std::ifstream;
using std::ios;
uint32_t reverseBytes(uint32_t n){
    return (n<<24) & (0xFF000000) | (n<<8) & (0x00FF0000) | (n>>8) & (0x0000FF00) | (n>>24) & (0x000000FF);
}

vector<Matrix> loadImages(string fileName){
    vector<Matrix> images;
    ifstream file(fileName, ios::binary);
    if(!file.is_open()){
        throw "Unable To Open file!";
    }
    else{
        uint32_t magic, numImages, rows, cols;
        file.read((char*)&magic, 4);
        file.read((char*)&numImages, 4);
        file.read((char*)&rows, 4);
        file.read((char*)&cols, 4);
        magic =reverseBytes(magic);
        numImages = reverseBytes(numImages);
        rows = reverseBytes(rows);
        cols = reverseBytes(cols);
        for (int i=0 ; i < (int)numImages ; i++){
            Matrix temp(784,1);
            for( int j=0 ; j < 784 ; j++ ){
                unsigned char pixel;
                file.read((char*)&pixel, 1);
                temp.set_element( j , 0 , pixel/255.0);
            }
            images.push_back(temp);
        }
    }
    file.close();
    return images;
}

vector<int> loadLabels(string fileName){
    vector<int> labels;
    ifstream file( fileName , ios::binary );
    if(!file.is_open()){
        throw "Unable To Open file!";
    }
    else{
        uint32_t magic, numLabels;
        file.read((char*)&magic, 4);
        file.read((char*)&numLabels, 4);
        magic = reverseBytes(magic);
        numLabels = reverseBytes(numLabels);
        for ( int i = 0 ; i < (int)numLabels ; i++){
            unsigned char label;
            file.read((char*)&label, 1);
            labels.push_back((int)label);
        }
    }
    file.close();
    return labels;
}