#include "ImageRecovery.h"
#include <iomanip>

int main(int argc, char **argv)
{
    cv::Mat source_img = imread(argv[1], cv::IMREAD_UNCHANGED);
    if (!source_img.data)
    {
        std::cout << "\n Image not found (wrong path) !";
        std::cout << "\n Path: " << argv[1];
        return 0;
    }

    cv::Mat degraded_img;
    bool recov = str_compare(argv[4], "-recov");
    if (recov)
    {
        degraded_img = imread(argv[2], cv::IMREAD_UNCHANGED);

        if (!degraded_img.data)
        {
            std::cout << "\n Image not found (wrong path) !";
            std::cout << "\n Path: " << argv[2];
            return 0;
        }
    }

    cv::Mat dest_img;
    bool success = false;

    // Degradation prior knowledge
    bool gau = false;
    double sigma = 0;
    bool motion = false;
    double angle = 0;
    bool defocus = false;
    double radius = 0;
    double size = 0;
    double noise_std = 0;
    bool use_src_ps = true;

    // Recovery approach
    bool ls = false, cls = false;
    double threshold_T = 0, alpha = 0;
    bool iterLS = false, iterCLS = false;
    double beta = 0;
    int maxIter = 0;
    bool weiner = false;
    bool blind = false;

    /* -<psf-type> <psf-size> -degr <sigma || angle || radius> <noise-std>*/
    if (str_compare(argv[3], "-degr"))
    {
        if (str_compare(argv[4], "-gau"))
        {
            success = applyDegradation(source_img, dest_img, createGaussianPSF(char_2_int(argv, 5), char_2_double(argv, 6)), char_2_double(argv, 7));
        }
        else if (str_compare(argv[4], "-motion"))
        {
            success = applyDegradation(source_img, dest_img, createMotionBlurPSF(char_2_int(argv, 5), char_2_double(argv, 6)), char_2_double(argv, 7));
        }
        else if (str_compare(argv[4], "-defocus"))
        {
            success = applyDegradation(source_img, dest_img, createDefocusPSF(char_2_int(argv, 5), char_2_double(argv, 6)), char_2_double(argv, 7));
        }
    }
    else if (str_compare(argv[4], "-recov"))
    {
        cv::Mat psf;

        if (str_compare(argv[6], "-gau"))
        {
            psf = createGaussianPSF(char_2_int(argv, 7), char_2_double(argv, 8));
            gau = true;
            size = char_2_int(argv, 7);
            sigma = char_2_double(argv, 8);
        }
        else if (str_compare(argv[6], "-motion"))
        {
            psf = createMotionBlurPSF(char_2_int(argv, 7), char_2_double(argv, 8));
            motion = true;
            size = char_2_int(argv, 7);
            angle = char_2_double(argv, 8);
        }
        else if (str_compare(argv[6], "-defocus"))
        {
            psf = createDefocusPSF(char_2_int(argv, 7), char_2_double(argv, 8));
            defocus = true;
            size = char_2_int(argv, 7);
            radius = char_2_double(argv, 8);
        }

        /* <source-path> <input-path> <output-path> -recov -ls -<psf-type> <psf-size> <sigma || angle || radius> <threshold-T>*/
        if (str_compare(argv[5], "-ls"))
        {
            ls = true;
            threshold_T = char_2_double(argv, 9);

            if (degraded_img.channels() == 3)
            {
                std::vector<cv::Mat> bgrChannels(3), restoredChannels(3);
                cv::split(degraded_img, bgrChannels);

                for (int i = 0; i < 3; i++)
                {
                    success = ImageRecovery::leastSquares(bgrChannels[i], restoredChannels[i], psf, threshold_T);
                }

                cv::merge(restoredChannels, dest_img);
            }
            else
            {
                success = ImageRecovery::leastSquares(degraded_img, dest_img, psf, threshold_T);
            }
        }
        /* <source-path> <input-path> <output-path> -recov -iterLS -<psf-type> <psf-size> <sigma || angle || radius> <beta> <maxIter> */
        else if (str_compare(argv[5], "-iterLS"))
        {
            iterLS = true;
            beta = char_2_double(argv, 9);
            maxIter = char_2_int(argv, 10);

            if (degraded_img.channels() == 3)
            {
                std::vector<cv::Mat> bgrChannels(3), restoredChannels(3);
                cv::split(degraded_img, bgrChannels);

                for (int i = 0; i < 3; i++)
                {
                    success = ImageRecovery::iterativeLeastSquares(bgrChannels[i], restoredChannels[i], psf, beta, maxIter);
                }

                cv::merge(restoredChannels, dest_img);
            }
            else
            {
                success = ImageRecovery::iterativeLeastSquares(degraded_img, dest_img, psf, beta, maxIter);
            }
        }
        /* <source-path> <degraded-path> <output-path> -recov -cls -<psf-type> <psf-size> <sigma || angle || radius> <alpha> <threshold-T> */
        else if (str_compare(argv[5], "-cls"))
        {
            cls = true;
            alpha = char_2_double(argv, 9);
            threshold_T = char_2_double(argv, 10);

            if (degraded_img.channels() == 3)
            {
                std::vector<cv::Mat> bgrChannels(3), restoredChannels(3);
                cv::split(degraded_img, bgrChannels);

                for (int i = 0; i < 3; i++)
                {
                    success = ImageRecovery::iterativeLeastSquares(bgrChannels[i], restoredChannels[i], psf, beta, maxIter);
                }

                cv::merge(restoredChannels, dest_img);
            }
            else
            {
                success = ImageRecovery::constrainedLS(degraded_img, dest_img, psf, alpha, threshold_T);
            }
        }
        /* <source-path> <degraded-path> <output-path> -recov -iterCLS -<psf-type> <psf-size> <sigma || angle || radius> <alpha> <beta> <maxIter> */
        else if (str_compare(argv[5], "-iterCLS"))
        {
            iterCLS = true;
            alpha = char_2_double(argv, 9);
            beta = char_2_double(argv, 10);
            maxIter = char_2_int(argv, 11);

            if (degraded_img.channels() == 3)
            {
                std::vector<cv::Mat> bgrChannels(3), restoredChannels(3);
                cv::split(degraded_img, bgrChannels);

                for (int i = 0; i < 3; i++)
                {
                    success = ImageRecovery::iterativeCLS(bgrChannels[i], restoredChannels[i], psf, alpha, beta, maxIter);
                }

                cv::merge(restoredChannels, dest_img);
            }
            else
            {
                success = ImageRecovery::iterativeCLS(degraded_img, dest_img, psf, alpha, beta, maxIter);
            }
        }
        /* <source-path> <degraded-path> <output-path> -recov -weiner -<psf-type> <psf-size> <sigma || angle || radius> <alpha> <noise-var> */
        else if (str_compare(argv[5], "-weiner"))
        {
            weiner = true;
            noise_std = char_2_double(argv, 9);
            use_src_ps = char_2_double(argv, 10) == 1;

            if (degraded_img.channels() == 3)
            {
                std::vector<cv::Mat> srcChannels(3), bgrChannels(3), restoredChannels(3);
                cv::split(source_img, srcChannels);
                cv::split(degraded_img, bgrChannels);

                for (int i = 0; i < 3; i++)
                {
                    success = ImageRecovery::weinerFilter(srcChannels[i], bgrChannels[i], restoredChannels[i], psf, noise_std, use_src_ps);
                }

                cv::merge(restoredChannels, dest_img);
            }
            else
            {
                success = ImageRecovery::weinerFilter(source_img, degraded_img, dest_img, psf, noise_std, use_src_ps);
            }
        }
    }

    if (success)
    {
        imshow("Source image", source_img);
        imshow("Processed image", dest_img);

        if (recov)
        {
            imshow("Degraded image", degraded_img);

            double isnr = 0;
            double psnr = 0;

            if (degraded_img.channels() == 3)
            {
                std::vector<cv::Mat> origChannels(3), degrChannels(3), restChannels(3);
                cv::split(source_img, origChannels);
                cv::split(degraded_img, degrChannels);
                cv::split(dest_img, restChannels);

                for (int i = 0; i < 3; ++i)
                {
                    isnr += computeISNR(origChannels[i], degrChannels[i], restChannels[i]);
                    psnr += computePSNR(origChannels[i], restChannels[i]);
                }

                isnr /= 3;
                psnr /= 3;
            }
            else
            {
                isnr = computeISNR(source_img, degraded_img, dest_img);
                psnr = computePSNR(source_img, dest_img);
            }

            if (ls)
            {
                std::cout << "\n Least Squares" << std::endl;
                std::cout << "\n Threshold T: " << threshold_T;
            }
            else if (iterLS)
            {
                std::cout << "\n Iterative Least Squares" << std::endl;
                std::cout << "\n Beta: " << beta;
                std::cout << "\n Max Iteration: " << maxIter;
            }
            else if (iterCLS)
            {
                std::cout << "\n Iterative Constrained Least Squares";
            }
            else if (cls)
            {
                std::cout << "\n Constrained Least Squares" << std::endl;
                std::cout << "\n Alpha for regularization term Cf: " << alpha;
                std::cout << "\n Threshold T: " << threshold_T;
            }
            else if (weiner)
            {
                std::cout << "\n Weiner Restoration Filter" << std::endl;
                std::cout << "\n Noise Standard Deviation: " << noise_std;
            }
            else if (blind)
            {
                std::cout << "\n Blind Restoration";
            }
            std::cout << std::endl;

            std::cout << "\n Degradation Prior Knowledge: ";
            if (gau)
            {
                std::cout << "\n --> Gaussian Blur";
                std::cout << "\n --> Size: " << size;
                std::cout << "\n --> Sigma: " << sigma;
            }
            else if (motion)
            {
                std::cout << "\n --> Motion Blur";
                std::cout << "\n --> Size: " << size;
                std::cout << "\n --> Angle: " << angle;
            }
            else if (defocus)
            {
                std::cout << "\n --> Defocusing";
                std::cout << "\n --> Size: " << size;
                std::cout << "\n --> Radius: " << radius;
            }
            std::cout << std::endl;

            std::cout << "\n ISNR: " << isnr;
            std::cout << "\n PSNR: " << psnr << std::endl;
            std::cout << std::endl;
        }

        if (recov)
            imwrite(argv[3], dest_img);
        else
            imwrite(argv[2], dest_img);
    }
    else
        std::cout << "\n Something went wrong!";

    cv::waitKey(0);
    return 0;
}
