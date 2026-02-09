#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "OpenCV version: "
              << CV_VERSION << std::endl;

    cv::Mat img = cv::Mat::zeros(240, 320, CV_8UC3);

    std::cout << "Imagen creada: "
              << img.cols << "x" << img.rows << std::endl;

    std::cout << "OpenCV CPU OK ✅" << std::endl;
    return 0;
}
