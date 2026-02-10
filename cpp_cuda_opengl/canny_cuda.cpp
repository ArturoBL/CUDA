#include <opencv2/opencv.hpp>

// CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>
#include <chrono>

/*
 * Compilar con:
   g++ canny_cuda.cpp -o canny_cuda   `pkg-config --cflags --libs opencv4`   -I/usr/local/cuda/include   -L/usr/local/cuda/lib64   -lcuda -lcudart
*/   

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Uso: ./canny_cuda video.mp4\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cerr << "No se pudo abrir el video\n";
        return -1;
    }

    cv::Mat frame, gray_cpu, edges_cpu;
    cv::cuda::GpuMat d_frame, d_gray, d_edges;

    cv::Ptr<cv::cuda::CannyEdgeDetector> canny =
        cv::cuda::createCannyEdgeDetector(50.0, 150.0);

    cv::cuda::Stream stream;

    int frames = 0;
    double cpu_ms_total = 0.0;
    double gpu_ms_total = 0.0;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // ================= CPU =================
        auto cpu_start = std::chrono::high_resolution_clock::now();

        cv::cvtColor(frame, gray_cpu, cv::COLOR_BGR2GRAY);
        cv::Canny(gray_cpu, edges_cpu, 50, 150);

        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_ms_total += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // ================= GPU =================
        auto gpu_start = std::chrono::high_resolution_clock::now();

        d_frame.upload(frame, stream);
        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY, 0, stream);
        canny->detect(d_gray, d_edges, stream);

        stream.waitForCompletion();

        auto gpu_end = std::chrono::high_resolution_clock::now();
        gpu_ms_total += std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        frames++;

        // Solo para visualizar (opcional)
        d_edges.download(edges_cpu);
        cv::imshow("Canny CUDA", edges_cpu);

        if (cv::waitKey(1) == 27)
            break;
    }

    std::cout << "\n===== RESULTADOS =====\n";
    std::cout << "Frames procesados: " << frames << "\n";
    std::cout << "CPU promedio FPS : " << (frames * 1000.0 / cpu_ms_total) << "\n";
    std::cout << "GPU promedio FPS : " << (frames * 1000.0 / gpu_ms_total) << "\n";

    return 0;
}

