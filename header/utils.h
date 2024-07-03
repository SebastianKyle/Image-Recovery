#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "libs.h"

cv::Mat computeDFT(const cv::Mat& source_img, const cv::Size &size = cv::Size());
cv::Mat computeIDFT(const cv::Mat& cplx_img);
cv::Mat applyConvolution(const cv::Mat& source_img, const cv::Mat& kernel);
cv::Mat createGaussianPSF(int size, double sigma);
cv::Mat createMotionBlurPSF(int size, double angle);
cv::Mat createDefocusPSF(int size, double radius);
int applyDegradation(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &pdf, float noise_std);

cv::Mat conjugate(const cv::Mat &cplx_mat);

double computeRMSE(const cv::Mat &img1, const cv::Mat &img2);
double computeISNR(const cv::Mat &original_img, const cv::Mat &degraded_img, const cv::Mat &restored_img);
double computePSNR(const cv::Mat &original_img, const cv::Mat &restored_img);

#endif