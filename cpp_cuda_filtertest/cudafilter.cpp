#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>

int main()
{
    auto f = cv::cuda::createGaussianFilter(
        CV_8UC3,
        CV_8UC3,
        cv::Size(5,5),
        1.5
    );
    return 0;
}
