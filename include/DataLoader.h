#pragma once
#include <string>
#include <vector>
#include <cstdint>         // for uint32_t (guaranteed 32-bit unsigned integer)
#include "Matrix.h"

using std::string;
using std::vector;

// reverseBytes
// MNIST files store integers in big-endian format (most significant byte first).
// Most modern CPUs (x86/x64) are little-endian (least significant byte first).
// This function swaps the byte order of a 32-bit integer to convert between them.
//
// Example: big-endian 0x0000EA60 (60,000) stored as bytes: 00 00 EA 60
//          After reverseBytes it becomes: 60 EA 00 00 → read as 0x0000EA60 ✓
uint32_t reverseBytes(uint32_t n);

// loadImages
// Reads the MNIST image binary file and returns a vector of Matrix objects,
// one per image. Each Matrix is (784×1) — a flattened 28×28 pixel grid stored
// as a column vector, with pixel values normalized from [0,255] to [0.0,1.0].
// Normalization helps the network train faster with smaller weight values.
vector<NN::Matrix> loadImages(string fileName);

// loadLabels
// Reads the MNIST label binary file and returns a vector of integers (0-9),
// one label per image. Labels are used to construct one-hot target vectors
// during training via the oneHot() function in main.cpp.
vector<int> loadLabels(string fileName);