#pragma once
#ifndef IMAGERECOVERY_H
#define IMAGERECOVERY_H

#include "utils.h"

class ImageRecovery
{
private:
    static double computeTV(const cv::Mat &img, const cv::Mat &u); // Compute total variation
    static cv::Mat updateX(const cv::Mat &y, const cv::Mat &h, const cv::Mat &u, double beta, double alpha_im, int iters, double learning_rate);
    static cv::Mat updateH(const cv::Mat &y, const cv::Mat &x, double beta, double alpha_bl, int iters, double learning_rate);
    static cv::Mat updateU(const cv::Mat &x);
    static void updateHyperparameters(double &alpha_im, double &alpha_bl, double &beta, const cv::Mat &x, const cv::Mat &h);
    static cv::Mat initializePSF(int rows, int cols, const std::string& method="uniform");

public:
    static int leastSquares(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double threshold_T);
    static int iterativeLeastSquares(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double beta, int maxIter);

    static int constrainedLS(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double alpha, double threshold_T);
    static int iterativeCLS(const cv::Mat &degraded_img, cv::Mat &dest_img, const cv::Mat &psf, double alpha, double beta, int maxIter);

    static cv::Mat estimatePowerSpectrum(const cv::Mat &img);
    static int weinerFilter(const cv::Mat &source_img, const cv::Mat &degraded_img, cv::Mat &dest_img, const cv::Mat &psf, double noise_std, bool use_src_ps = true);

    static int variationalBayesianInference(const cv::Mat &degraded_img, cv::Mat &dest_img, int psf_rows, int psf_cols, int iters, double learning_rate);
};

#endif