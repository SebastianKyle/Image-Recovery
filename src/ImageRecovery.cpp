#include "ImageRecovery.h"

int ImageRecovery::leastSquares(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double threshold_T)
{
    if (!source_img.data)
        return 0;

    cv::Mat source_float;
    source_img.convertTo(source_float, CV_32F);

    // Y(u, v)
    cv::Mat Y = computeDFT(source_float);
    cv::Mat H = computeDFT(psf, source_img.size());

    cv::Mat H_conj;
    cv::mulSpectrums(cv::Mat::ones(H.rows, H.cols, H.type()), H, H_conj, 0, true);

    cv::Mat H_mag_squared;
    cv::mulSpectrums(H, H_conj, H_mag_squared, 0);

    cv::Mat H_mag_squared_inv = H_mag_squared.clone();
    H_mag_squared_inv += cv::Scalar::all(1e-6);
    cv::divide(1.0, H_mag_squared_inv, H_mag_squared_inv);

    double max_threshold_value = 1 / threshold_T;
    cv::threshold(H_mag_squared_inv, H_mag_squared_inv, max_threshold_value, 0.0, cv::THRESH_TOZERO_INV);

    cv::Mat F;
    cv::mulSpectrums(H_conj, Y, F, 0);
    cv::mulSpectrums(F, H_mag_squared_inv, F, 0);

    dest_img = computeIDFT(F);

    cv::normalize(dest_img, dest_img, 0, 255, cv::NORM_MINMAX);
    dest_img.convertTo(dest_img, CV_8U);

    return 1;
}

int ImageRecovery::iterativeLeastSquares(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double beta, int maxIter)
{
    if (!source_img.data)
        return 0;

    cv::Mat source_float;
    source_img.convertTo(source_float, CV_32F);

    // Y(u, v)
    cv::Mat Y = computeDFT(source_float);
    cv::Mat H = computeDFT(psf, source_img.size());

    cv::Mat H_conj = conjugate(H);

    cv::Mat H_mag_squared;
    cv::mulSpectrums(H, H_conj, H_mag_squared, 0);

    cv::Mat planes[2];
    cv::split(H_mag_squared, planes);
    double max_H = 0;
    cv::Point max_loc;
    cv::minMaxLoc(planes[0], nullptr, &max_H, nullptr, &max_loc);
    std::cout << "\n Upper range for beta: " << (double) 2/max_H << std::endl;

    // Initialize F0
    cv::Mat F = cv::Mat::zeros(Y.size(), Y.type());

    double prev_rmse = std::numeric_limits<double>::max();
    for (int iter = 0; iter < maxIter; iter++)
    {
        // beta.H*(u,v).Y(u,v)
        cv::Mat term1;
        cv::mulSpectrums(H_conj, Y, term1, 0);
        term1 *= beta;

        // (1 - beta.|H(u, v)|^2).Fk(u, v)
        cv::Mat term2;
        cv::Mat one_minus_beta_H_mag_squared;
        cv::subtract(cv::Scalar::all(1.0), H_mag_squared * beta, one_minus_beta_H_mag_squared);
        cv::mulSpectrums(one_minus_beta_H_mag_squared, F, term2, 0);

        // Obtain Fk+1
        cv::Mat F_new;
        cv::add(term1, term2, F_new);

        cv::Mat spatial_F_new = computeIDFT(F_new);
        cv::Mat spatial_F = computeIDFT(F);
        double current_rmse = computeRMSE(spatial_F_new, spatial_F);

        F = F_new;

        // Check for convergence
        if (current_rmse < 1e-4)
        {
            break;
        }
        prev_rmse = current_rmse;
    }

    dest_img = computeIDFT(F);

    cv::normalize(dest_img, dest_img, 0, 255, cv::NORM_MINMAX);
    dest_img.convertTo(dest_img, CV_8U);

    return 1;
}

int ImageRecovery::constrainedLS(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double alpha, double threshold_T) {
    if (!source_img.data)
        return 0;

    cv::Mat source_float;
    source_img.convertTo(source_float, CV_32F);

    cv::Mat Y = computeDFT(source_float);
    cv::Mat H = computeDFT(psf, source_float.size());

    cv::Mat laplacian = (cv::Mat_<float>(3, 3) << 0, 0.25, 0, 0.25, -1, 0.25, 0, 0.25, 0);
    cv::Mat C = computeDFT(laplacian, source_float.size());

    cv::Mat H_conj = conjugate(H);
    cv::Mat H_mag_squared;
    cv::mulSpectrums(H, H_conj, H_mag_squared, 0);

    cv::Mat C_mag_2;
    cv::magnitude(C, C, C_mag_2);
    C_mag_2 = C_mag_2.mul(C_mag_2);

    cv::Mat denominator = H_mag_squared + alpha * C_mag_2;
    denominator += cv::Scalar::all(1e-6);
    cv::Mat denominator_inv;
    cv::divide(1.0, denominator, denominator_inv);

    double max_threshold = 1 / threshold_T;
    cv::threshold(denominator_inv, denominator_inv, max_threshold, 0.0, cv::THRESH_TOZERO_INV);

    cv::Mat F;
    cv::mulSpectrums(Y, H_conj, F, 0);
    cv::mulSpectrums(F, denominator_inv, F, 0);

    dest_img = computeIDFT(F);

    cv::normalize(dest_img, dest_img, 0, 255, cv::NORM_MINMAX);
    dest_img.convertTo(dest_img, CV_8U);

    return 1;
}

int ImageRecovery::iterativeCLS(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, int maxIter)
{
    if (!source_img.data)
        return 0;

    return 1;
}

int ImageRecovery::weinerFilter(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double noiseVariance)
{
    if (!source_img.data)
        return 0;

    return 1;
}

int ImageRecovery::blindDeconvolution(const cv::Mat &source_img, cv::Mat &dest_img, int iterations)
{
    if (!source_img.data)
        return 0;

    return 1;
}