#include "ImageRecovery.h"

int ImageRecovery::leastSquares(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double threshold_T)
{
    if (!source_img.data)
        return 0;

    cv::Mat source_float;
    source_img.convertTo(source_float, CV_32F);

    // Y(u, v)
    cv::Mat Y = computeDFT(source_float);
    cv::Mat H = computeDFT(psf, Y.size());

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

    dest_img = computeIDFT(F, source_float.size());

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
    cv::Mat H = computeDFT(psf, Y.size());

    cv::Mat H_conj = conjugate(H);

    cv::Mat H_mag_squared;
    cv::mulSpectrums(H, H_conj, H_mag_squared, 0);

    // Initialize F0
    cv::Mat F = cv::Mat::zeros(Y.size(), Y.type());

    /* Compute convergence term Rk(u, v) */
    cv::Mat beta_H_mag_2 = H_mag_squared * beta;
    cv::Mat numerator = cv::Mat::ones(H.rows, H.cols, H.type());
    cv::Mat one_minus_beta_H_mag_2;
    cv::subtract(cv::Scalar::all(1.0), beta_H_mag_2, one_minus_beta_H_mag_2);

    cv::Mat denominator_inv = beta_H_mag_2.clone();
    denominator_inv += cv::Scalar::all(1e-6);
    cv::divide(1.0, denominator_inv, denominator_inv);
    double max_threshold = 1e5;
    cv::threshold(denominator_inv, denominator_inv, max_threshold, 0.0, cv::THRESH_TOZERO_INV);

    double min_threshold = 1e-20;
    for (int iter = 1; iter < maxIter; iter++)
    {
        cv::mulSpectrums(numerator, one_minus_beta_H_mag_2, numerator, 0);
        cv::max(numerator, min_threshold, numerator);
    }
    cv::subtract(cv::Scalar::all(1.0), numerator, numerator);

    cv::Mat Rk;
    cv::mulSpectrums(numerator, denominator_inv, Rk, 0);
    cv::mulSpectrums(Rk, H_conj, Rk, 0);
    Rk *= beta;

    cv::mulSpectrums(Rk, Y, F, 0);

    /* ---------------------------------- */

    dest_img = computeIDFT(F, source_float.size());

    cv::normalize(dest_img, dest_img, 0, 255, cv::NORM_MINMAX);
    dest_img.convertTo(dest_img, CV_8U);

    return 1;
}

int ImageRecovery::constrainedLS(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, double alpha, double threshold_T)
{
    if (!source_img.data)
        return 0;

    cv::Mat source_float;
    source_img.convertTo(source_float, CV_32F);

    cv::Mat Y = computeDFT(source_float);
    cv::Mat H = computeDFT(psf, Y.size());

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

    dest_img = computeIDFT(F, source_float.size());

    cv::normalize(dest_img, dest_img, 0, 255, cv::NORM_MINMAX);
    dest_img.convertTo(dest_img, CV_8U);

    return 1;
}

int ImageRecovery::iterativeCLS(const cv::Mat &degraded_img, cv::Mat &dest_img, const cv::Mat &psf, double alpha, double beta, int maxIter)
{
    if (!degraded_img.data)
        return 0;

    cv::Mat source_float;
    degraded_img.convertTo(source_float, CV_32F);

    cv::Mat Y = computeDFT(source_float);
    cv::Mat H = computeDFT(psf, Y.size());

    cv::Mat H_conj = conjugate(H);

    cv::Mat H_mag_squared;
    cv::mulSpectrums(H, H_conj, H_mag_squared, 0);

    cv::Mat laplacian = (cv::Mat_<float>(3, 3) << 0, 0.25, 0, 0.25, -1, 0.25, 0, 0.25, 0);
    cv::Mat C = computeDFT(laplacian, Y.size());
    cv::Mat C_mag_2;
    cv::mulSpectrums(C, conjugate(C), C_mag_2, 0);

    cv::Mat F = cv::Mat::zeros(Y.size(), Y.type());

    /* Compute convergence term Rk(u, v) */

    cv::Mat H_Mag_C_Mag;
    cv::add(H_mag_squared, C_mag_2 * alpha, H_Mag_C_Mag);
    cv::Mat beta_H_mag_2_C_mag_2 = H_Mag_C_Mag * beta;
    cv::Mat numerator = cv::Mat::ones(H.rows, H.cols, H.type());
    cv::Mat one_minus_beta_H_mag_2_C_mag_2;
    cv::subtract(cv::Scalar::all(1.0), beta_H_mag_2_C_mag_2, one_minus_beta_H_mag_2_C_mag_2);

    cv::Mat denominator_inv = beta_H_mag_2_C_mag_2.clone();
    denominator_inv += cv::Scalar::all(1e-6);
    cv::divide(1.0, denominator_inv, denominator_inv);
    double max_threshold = 1e5;
    cv::threshold(denominator_inv, denominator_inv, max_threshold, 0.0, cv::THRESH_TOZERO_INV);

    double min_threshold = 1e-20;
    for (int iter = 1; iter < maxIter; iter++)
    {
        cv::mulSpectrums(numerator, one_minus_beta_H_mag_2_C_mag_2, numerator, 0);
        cv::max(numerator, min_threshold, numerator);
    }
    cv::subtract(cv::Scalar::all(1.0), numerator, numerator);

    cv::Mat Rk;
    cv::mulSpectrums(numerator, denominator_inv, Rk, 0);
    cv::mulSpectrums(Rk, H_conj, Rk, 0);
    Rk *= beta;

    cv::mulSpectrums(Rk, Y, F, 0);

    /* ---------------------------------- */

    dest_img = computeIDFT(F, source_float.size());

    cv::normalize(dest_img, dest_img, 0, 255, cv::NORM_MINMAX);
    dest_img.convertTo(dest_img, CV_8U);

    return 1;
}

cv::Mat ImageRecovery::estimatePowerSpectrum(const cv::Mat &img)
{
    cv::Mat dft_img = computeDFT(img);
    cv::Mat power_spectrum;
    cv::mulSpectrums(dft_img, dft_img, power_spectrum, 0, true);

    return power_spectrum;
}

int ImageRecovery::weinerFilter(const cv::Mat &source_img, const cv::Mat &degraded_img, cv::Mat &dest_img, const cv::Mat &psf, double noise_std, bool use_src_ps)
{
    if (!degraded_img.data)
        return 0;

    cv::Mat source_float;
    degraded_img.convertTo(source_float, CV_32F);

    cv::Mat Y = computeDFT(source_float);
    cv::Mat H = computeDFT(psf, Y.size());

    cv::Mat H_conj = conjugate(H);

    cv::Mat H_mag_squared;
    cv::mulSpectrums(H, H_conj, H_mag_squared, 0);

    // Estimate original image power spectrum
    cv::Mat S_f;
    if (use_src_ps)
    {
        S_f = estimatePowerSpectrum(source_img);
    }
    else
    {
        S_f = estimatePowerSpectrum(degraded_img);
    }
    cv::Mat S_f_inv = S_f.clone();
    S_f_inv += cv::Scalar::all(1e-6);
    cv::divide(1.0, S_f_inv, S_f_inv);

    double max_threshold = 1e5;

    cv::threshold(S_f_inv, S_f_inv, max_threshold, 0.0, cv::THRESH_TOZERO_INV);

    // Noise power spectrum
    cv::Mat noise = cv::Mat::zeros(degraded_img.size(), CV_32F);
    cv::randn(noise, 0, noise_std);
    cv::Mat noise_dft = computeDFT(noise);
    cv::Mat S_n;
    cv::mulSpectrums(noise_dft, noise_dft, S_n, 0, true);

    // Noise to signal power ratio (S_n / S_f)
    cv::Mat NSR;
    cv::mulSpectrums(S_n, S_f_inv, NSR, 0);

    cv::Mat denominator = H_mag_squared + NSR;
    cv::Mat denominator_inv = denominator.clone();
    denominator_inv += cv::Scalar::all(1e-6);
    cv::divide(1.0, denominator_inv, denominator_inv);
    cv::threshold(denominator_inv, denominator_inv, max_threshold, 0.0, cv::THRESH_TOZERO_INV);

    cv::Mat R;
    cv::mulSpectrums(H_conj, denominator_inv, R, 0);

    cv::Mat F;
    cv::mulSpectrums(R, Y, F, 0);

    dest_img = computeIDFT(F, source_float.size());

    cv::normalize(dest_img, dest_img, 0, 255, cv::NORM_MINMAX);
    dest_img.convertTo(dest_img, CV_8U);

    return 1;
}

double ImageRecovery::computeTV(const cv::Mat &img, const cv::Mat &u)
{
    double tv = 0.0;

    for (int y = 0; y < img.rows - 1; y++)
    {
        for (int x = 0; x < img.cols - 1; x++)
        {
            double dx = img.at<double>(y, x + 1) - img.at<double>(y, x);
            double dy = img.at<double>(y + 1, x) - img.at<double>(y, x);
            tv += u.at<double>(y, x) * std::sqrt(dx * dx + dy * dy + 1e-6);
        }
    }

    return tv;
}

cv::Mat ImageRecovery::updateX(const cv::Mat &y, const cv::Mat &h, const cv::Mat &u, double beta, double alpha_im, int iters, double learning_rate)
{
    cv::Mat x = cv::Mat::zeros(y.size(), CV_64F);
    cv::Mat Hx;

    for (int iter = 0; iter < iters; iter++)
    {
        cv::filter2D(x, Hx, -1, h, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::Mat grad = beta * (Hx - y);

        for (int i = 0; i < x.rows - 1; i++)
        {
            for (int j = 0; j < x.cols - 1; j++)
            {
                double dx = x.at<double>(i, j + 1) - x.at<double>(i, j);
                double dy = x.at<double>(i + 1, j) - x.at<double>(i, j);

                grad.at<double>(i, j) += alpha_im * u.at<double>(i, j) * (dx + dy);
            }
        }

        x -= learning_rate * grad;
    }

    return x;
}

cv::Mat ImageRecovery::updateH(const cv::Mat &y, const cv::Mat &x, double beta, double alpha_bl, int iters, double learning_rate)
{
    cv::Mat h = cv::Mat::ones(3, 3, CV_64F);
    cv::Mat Hx;

    for (int iter = 0; iter < iters; iter++)
    {
        cv::filter2D(x, Hx, -1, h, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::Mat grad = beta * (Hx - y);

        cv::Mat laplacian;
        cv::Laplacian(h, laplacian, -1);
        grad += alpha_bl * laplacian;
        h -= learning_rate * grad;
    }

    return h;
}

cv::Mat ImageRecovery::updateU(const cv::Mat &x)
{
    cv::Mat u = cv::Mat::ones(x.size(), CV_64F);

    for (int i = 0; i < x.rows - 1; ++i)
    {
        for (int j = 0; j < x.cols - 1; ++j)
        {
            double dx = x.at<double>(i, j + 1) - x.at<double>(i, j);
            double dy = x.at<double>(i + 1, j) - x.at<double>(i, j);
            u.at<double>(i, j) = 1.0 / std::sqrt(dx * dx + dy * dy + 1e-6);
        }
    }

    return u;
}

void ImageRecovery::updateHyperparameters(double &alpha_im, double &alpha_bl, double &beta, const cv::Mat &x, const cv::Mat &h)
{
    alpha_im = 1.0 / (0.5 * (computeTV(x, cv::Mat::ones(x.size(), CV_64F)) + 1e-6));
    alpha_bl = 1.0 / (0.5 * (cv::norm(h, cv::NORM_L2) + 1e-6));
    beta = 1.0 / (0.5 * (cv::norm(x - h, cv::NORM_L2) + 1e-6));
}

cv::Mat ImageRecovery::initializePSF(int rows, int cols, const std::string& method) {
    cv::Mat h = cv::Mat::ones(rows, cols, CV_64F);

    if (method == "gaussian") {
        cv::Mat gauss_kernel = cv::getGaussianKernel(rows, 1.0, CV_64F) * cv::getGaussianKernel(cols, 1.0, CV_64F).t();
        gauss_kernel.copyTo(h);
    }

    // Normalize
    h /= cv::sum(h)[0];

    return h;
}

int ImageRecovery::variationalBayesianInference(const cv::Mat &degraded_img, cv::Mat &dest_img, int psf_rows, int psf_cols, int iters, double learning_rate)
{
    if (!degraded_img.data)
        return 0;

    cv::Mat y;
    degraded_img.convertTo(y, CV_64F, 1.0 / 255.0);

    cv::Mat x = cv::Mat::zeros(y.size(), CV_64F);
    cv::Mat h = initializePSF(psf_rows, psf_cols, "gaussian");
    double alpha_im = 1.0, alpha_bl = 1.0, beta = 1.0;

    for (int iter = 0; iter < iters; iter++) {
        cv::Mat u = updateU(x);
        x = updateX(y, h, u, beta, alpha_im, iters, learning_rate);
        h = updateH(y, x, beta, alpha_bl, iters, learning_rate);
        updateHyperparameters(alpha_im, alpha_bl, beta, x, h);
    }

    x.convertTo(dest_img, CV_8U, 255.0);

    return 1;
}