#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

int main()
{
    // -----------------------------
    // Configuración
    // -----------------------------
    const int width  = 3840; // 4K
    const int height = 2160;
    const int iterations = 200;

    std::cout << "Resolucion: " << width << "x" << height << std::endl;
    std::cout << "Iteraciones: " << iterations << std::endl;

    if (cv::cuda::getCudaEnabledDeviceCount() == 0)
    {
        std::cerr << "No hay dispositivos CUDA disponibles\n";
        return -1;
    }

    cv::cuda::setDevice(0);
    cv::cuda::DeviceInfo device;
    std::cout << "GPU: " << device.name() << std::endl;

    // -----------------------------
    // Datos
    // -----------------------------
    cv::Mat src(height, width, CV_8UC3);
    cv::randu(src, 0, 255);

    cv::Mat cpu_dst;
    cv::cuda::GpuMat d_src, d_dst;

    // Upload UNA SOLA VEZ
    d_src.upload(src);

    // Crear filtro CUDA UNA SOLA VEZ
    auto gpu_filter = cv::cuda::createGaussianFilter(
        d_src.type(),
        d_src.type(),
        cv::Size(15, 15),
        1.5
    );

    // -----------------------------
    // Benchmark CPU
    // -----------------------------
    auto cpu_start = Clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        cv::GaussianBlur(src, cpu_dst, cv::Size(15, 15), 1.5);
    }

    auto cpu_end = Clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(
        cpu_end - cpu_start).count();

    // -----------------------------
    // Benchmark GPU (solo cómputo)
    // -----------------------------
    cv::cuda::Stream stream;

    auto gpu_start = Clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        gpu_filter->apply(d_src, d_dst, stream);
    }

    stream.waitForCompletion();

    auto gpu_end = Clock::now();

    double gpu_ms = std::chrono::duration<double, std::milli>(
        gpu_end - gpu_start).count();

    // -----------------------------
    // Resultados
    // -----------------------------
    double cpu_fps = iterations * 1000.0 / cpu_ms;
    double gpu_fps = iterations * 1000.0 / gpu_ms;

    std::cout << "\n=== RESULTADOS ===\n";
    std::cout << "CPU total: " << cpu_ms << " ms\n";
    std::cout << "CPU FPS:   " << cpu_fps << "\n\n";

    std::cout << "GPU total: " << gpu_ms << " ms\n";
    std::cout << "GPU FPS:   " << gpu_fps << "\n\n";

    std::cout << "Speedup GPU vs CPU: "
              << cpu_ms / gpu_ms << "x\n";

    return 0;
}
