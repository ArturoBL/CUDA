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

    // Verificar dispositivos CUDA
    int num_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (num_devices == 0) {
        std::cerr << "No hay GPU CUDA disponible." << std::endl;
        return -1;
    }

    cv::cuda::setDevice(0);

    cv::Mat frame;
    cv::cuda::GpuMat d_frame, d_blur;
    cv::Mat result;

    // Crear filtro gaussiano en GPU
    cv::Ptr<cv::cuda::Filter> gaussian =
        cv::cuda::createGaussianFilter(
            CV_8UC3, CV_8UC3,
            cv::Size(15, 15), 3
        );

    while (true) {
        if (!cap.read(frame))
            break;

        // Subir a GPU
        d_frame.upload(frame);

        // Aplicar blur en GPU
        gaussian->apply(d_frame, d_blur);

        // Descargar resultado
        d_blur.download(result);

        cv::imshow("Gaussian CUDA", result);

        if (cv::waitKey(1) == 27)
            break;
    }

    return 0;
}

