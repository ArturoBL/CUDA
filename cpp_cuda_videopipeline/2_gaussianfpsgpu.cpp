#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

int main() {
    std::string video_path = "video.mp4";
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el video." << std::endl;
        return -1;
    }

    cv::cuda::setDevice(0);

    cv::Mat frame;
    cv::cuda::GpuMat d_frame, d_blur;
    cv::Mat result;

    cv::Ptr<cv::cuda::Filter> gaussian =
        cv::cuda::createGaussianFilter(
            CV_8UC3, CV_8UC3,
            cv::Size(15, 15), 3
        );

    cv::TickMeter tm;
    int frame_count = 0;

    while (true) {
        if (!cap.read(frame))
            break;

        tm.start();

        d_frame.upload(frame);
        gaussian->apply(d_frame, d_blur);
        d_blur.download(result);

        tm.stop();
        frame_count++;

        cv::imshow("Gaussian CUDA", result);

        if (cv::waitKey(1) == 27)
            break;
    }

    double total_time = tm.getTimeSec();
    double fps = frame_count / total_time;

    std::cout << "Frames procesados: " << frame_count << std::endl;
    std::cout << "Tiempo total: " << total_time << " s" << std::endl;
    std::cout << "FPS promedio: " << fps << std::endl;

    return 0;
}

