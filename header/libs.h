#pragma once

#ifndef LIB_H
#define LIB_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <queue>
#include <omp.h>
#include <unordered_map>
#include <set>

bool str_compare(const char* a, std::string b);
double char_2_double(char* argv[], int n);
int char_2_int(char* argv[], int n);

#endif