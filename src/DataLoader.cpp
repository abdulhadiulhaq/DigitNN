// IDX FILE FORMAT:
//   Images file (idx3-ubyte):
//     Offset 0:  magic number (4 bytes) = 0x00000803
//     Offset 4:  number of images (4 bytes)
//     Offset 8:  number of rows (4 bytes) = 28
//     Offset 12: number of cols (4 bytes) = 28
//     Offset 16: pixel data — 1 byte per pixel, row by row, image by image
//
//   Labels file (idx1-ubyte):
//     Offset 0: magic number (4 bytes) = 0x00000801
//     Offset 4: number of labels (4 bytes)
//     Offset 8: label data — 1 byte per label (0-9)
#include "DataLoader.h"
#include <fstream>
#include "Matrix.h"
using std::ifstream;
using std::ios;

// reverseBytes
// Converts a 32-bit integer from big-endian to little-endian (or vice versa).
//
// Bit manipulation breakdown for n = 0xAABBCCDD:
//   (n << 24) & 0xFF000000  → 0xDD000000  (byte 3 moves to position 0)
//   (n <<  8) & 0x00FF0000  → 0x00CC0000  (byte 2 moves to position 1)
//   (n >>  8) & 0x0000FF00  → 0x0000BB00  (byte 1 moves to position 2)
//   (n >> 24) & 0x000000FF  → 0x000000AA  (byte 0 moves to position 3)
//   OR all together         → 0xDDCCBBAA  ✓ bytes fully reversed
uint32_t reverseBytes(uint32_t n){
    return (n<<24) & (0xFF000000) | (n<<8) & (0x00FF0000) | (n>>8) & (0x0000FF00) | (n>>24) & (0x000000FF);
}

// loadImages
// Opens the binary image file and reads all images into Matrix objects.
//
// Each image is 28×28 = 784 pixels. We flatten it into a (784×1) column vector
// because our network's first layer expects a 1D input column vector.
// The flattening is row-by-row: pixel[r][c] → index r*28 + c.
//
// Pixel normalization: divide by 255.0 to get values in [0.0, 1.0].
// This is important because:
//   - Large input values cause large pre-activations (Z = W*x + b)
//   - Large Z values push sigmoid to saturation (output near 0 or 1)
//   - Saturated sigmoid has near-zero gradient → very slow learning
//   - Normalized inputs keep activations in the sensitive range
//
// Uses ios::binary to prevent text-mode interpretation of byte values
// (e.g., 0x0A is the newline character on Windows and would be skipped).
vector<NN::Matrix> loadImages(string fileName){
    vector<NN::Matrix> images;
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
            NN::Matrix temp(784,1);
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

// loadLabels
// Opens the binary label file and reads all labels as integers.
//
// Each label is a single byte with value 0-9 representing the digit.
// We cast to int for convenient use in the training loop.
//
// Labels are NOT used directly as network targets. Instead, main.cpp converts
// each label to a one-hot vector via oneHot(label):
//   label=3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]^T
// This gives the network a clear target: output neuron 3 should be 1.0,
// all others should be 0.0.
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