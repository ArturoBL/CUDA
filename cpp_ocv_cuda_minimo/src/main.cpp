#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

int main() {
    int gpuCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "GPUs CUDA disponibles: " << gpuCount << std::endl;

    if (gpuCount == 0) {
        std::cerr << "OpenCV fue compilado SIN soporte CUDA\n";
        return -1;
    }

    cv::cuda::setDevice(0);

    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC1);

    cv::cuda::GpuMat d_img, d_blur;
    d_img.upload(img);

    // Gaussian Blur CUDA (forma correcta)
    cv::Ptr<cv::cuda::Filter> gauss =
        cv::cuda::createGaussianFilter(
            d_img.type(),
            d_img.type(),
            cv::Size(7, 7),
            1.5
        );

    gauss->apply(d_img, d_blur);

    cv::Mat result;
    d_blur.download(result);

    std::cout << "Filtro Gaussian CUDA ejecutado correctamente 🚀\n";
    return 0;
}
