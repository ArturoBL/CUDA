#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string video_path = "video.mp4";

    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo abrir el video." << std::endl;
        return -1;
    }

    std::cout << "Video abierto correctamente." << std::endl;

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Resolución: " << width << "x" << height << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Total frames: " << total << std::endl;

    cv::Mat frame;

    while (true) {
        if (!cap.read(frame)) {
            std::cout << "Fin del video." << std::endl;
            break;
        }

        cv::imshow("Video CPU Test", frame);

        if (cv::waitKey(30) == 27)  // ESC para salir
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

