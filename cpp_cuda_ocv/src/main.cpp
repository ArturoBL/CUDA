#include <iostream>
#include <opencv2/opencv.hpp>

extern "C" void launch_kernel();

int main() {
    std::cout << "Probando OpenCV + CUDA toolchain\n";

    launch_kernel();

    cv::Mat img = cv::Mat::zeros(240, 320, CV_8UC1);
    std::cout << "Imagen creada: "
              << img.cols << "x" << img.rows << std::endl;

    std::cout << "OpenCV OK ✅" << std::endl;
    return 0;
}

