#include "utils.h"

cv::Mat computeDFT(const cv::Mat &source_img, const cv::Size &size)
{
    cv::Mat padded;

    if (size.width != 0 && size.height != 0)
    {
        cv::copyMakeBorder(source_img, padded, 0, size.height - source_img.rows, 0, size.width - source_img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    }
    else
    {
        int m = cv::getOptimalDFTSize(source_img.rows);
        int n = cv::getOptimalDFTSize(source_img.cols);
        cv::copyMakeBorder(source_img, padded, 0, m - source_img.rows, 0, n - source_img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    }

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_img;
    cv::merge(planes, 2, complex_img);

    cv::dft(complex_img, complex_img);

    return complex_img;
}

cv::Mat computeIDFT(const cv::Mat &cplx_img)
{
    cv::Mat inverse_transform;
    cv::idft(cplx_img, inverse_transform, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    cv::Mat planes[2];
    cv::split(inverse_transform, planes);

    return planes[0];
}

cv::Mat applyConvolution(const cv::Mat &source_img, const cv::Mat &kernel)
{
    cv::Mat result;
    cv::filter2D(source_img, result, CV_32F, kernel);
    return result;
}

cv::Mat createGaussianPSF(int size, double sigma)
{
    cv::Mat psf(size, size, CV_32F);
    int half_size = size / 2;
    double sum = 0.0;

    for (int i = -half_size; i <= half_size; i++)
    {
        for (int j = -half_size; j <= half_size; j++)
        {
            double value = std::exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            psf.at<float>(i + half_size, j + half_size) = static_cast<float>(value);
            sum += value;
        }
    }

    psf /= static_cast<float>(sum);
    return psf;
}

cv::Mat createMotionBlurPSF(int size, double angle)
{
    cv::Mat psf = cv::Mat::zeros(size, size, CV_32F);
    int half_size = size / 2;
    double radians = angle * CV_PI / 180.0;

    int x = half_size;
    int y = half_size;

    for (int i = -half_size; i <= half_size; i++)
    {
        int new_x = static_cast<int>(half_size + i * std::cos(radians));
        int new_y = static_cast<int>(half_size + i * std::sin(radians));
        if (new_x >= 0 && new_x < size && new_y >= 0 && new_y < size)
        {
            psf.at<float>(new_y, new_x) = 1.0f;
        }
    }

    psf /= static_cast<float>(cv::countNonZero(psf));
    return psf;
}

cv::Mat createDefocusPSF(int size, double radius)
{
    cv::Mat psf = cv::Mat::zeros(size, size, CV_32F);
    int half_size = size / 2;
    double radius2 = radius * radius;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            double x = i - half_size;
            double y = j - half_size;

            if (x * x + y * y <= radius2)
            {
                psf.at<float>(i, j) = 1.0;
            }
        }
    }

    psf /= cv::sum(psf)[0];

    return psf;
}

int applyDegradation(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &psf, float noise_std)
{
    if (!source_img.data)
        return 0;

    cv::Mat source_float;
    source_img.convertTo(source_float, CV_32F);

    cv::Mat source_dft = computeDFT(source_float);
    cv::Mat psf_dft = computeDFT(psf, source_float.size());

    cv::Mat filtered_dft_img;
    cv::mulSpectrums(source_dft, psf_dft, filtered_dft_img, 0);

    cv::Mat filtered_img;
    filtered_img = computeIDFT(filtered_dft_img);

    cv::normalize(filtered_img, filtered_img, 0, 255, cv::NORM_MINMAX);
    filtered_img.convertTo(filtered_img, CV_8U);

    cv::Mat noise(filtered_img.size(), filtered_img.type());
    cv::randn(noise, 0, noise_std);
    filtered_img += noise;

    dest_img = filtered_img.clone();

    return 1;
}

cv::Mat conjugate(const cv::Mat &cplx_mat)
{
    cv::Mat mat_conj;
    cv::mulSpectrums(cv::Mat::ones(cplx_mat.rows, cplx_mat.cols, cplx_mat.type()), cplx_mat, mat_conj, 0, true);

    return mat_conj;
}

double computeRMSE(const cv::Mat &img1, const cv::Mat &img2)
{
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    cv::Scalar s = cv::sum(diff);
    double rmse = s[0] / (double)(img1.total());

    return std::sqrt(rmse);
}

double computeISNR(const cv::Mat &original_img, const cv::Mat &degraded_img, const cv::Mat &restored_img)
{
    cv::Mat orig_float, degraded_float, restored_float;
    original_img.convertTo(orig_float, CV_32F);
    degraded_img.convertTo(degraded_float, CV_32F);
    restored_img.convertTo(restored_float, CV_32F);

    cv::Mat diff_orig_degraded = orig_float - degraded_float;
    cv::Mat diff_orig_restored = orig_float - restored_float;

    double norm_orig_degraded = cv::norm(diff_orig_degraded, cv::NORM_L2SQR);
    double norm_orig_restored = cv::norm(diff_orig_restored, cv::NORM_L2SQR);

    double isnr = 10 * std::log10(norm_orig_degraded / norm_orig_restored);

    return isnr;
}

double computePSNR(const cv::Mat &original_img, const cv::Mat &restored_img)
{
    cv::Mat s1;
    cv::absdiff(original_img, restored_img, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);

    cv::Scalar s = cv::sum(s1);

    double sse = s.val[0] + s.val[1] + s.val[2];

    if (sse <= 1e-10)
        return 0;

    double mse = sse / (double)(original_img.channels() * original_img.total());
    double psnr = 10.0 * std::log10((255 * 255) / mse);

    return psnr;
}