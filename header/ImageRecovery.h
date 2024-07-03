#pragma once
#ifndef IMAGERECOVERY_H
#define IMAGERECOVERY_H

#include "utils.h"

class ImageRecovery
{
private:

public:
    static int leastSquares(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double threshold_T);
    static int iterativeLeastSquares(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double beta, int maxIter);
    static int constrainedLS(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double alpha, double threshold_T);
    static int iterativeCLS(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, int maxIter);
    static int weinerFilter(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double noiseVariance);
    static int blindDeconvolution(const cv::Mat &source_img, cv::Mat &dest_img, int iterations);
};

#endif