#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "Matrix.h"
using std::string;
using std::vector;

uint32_t reverseBytes(uint32_t n);
vector<Matrix> loadImages(string fileName);
vector<int> loadLabels(string fileName);