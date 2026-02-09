#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>

//Compilar con: g++ cpu_vs_cuda.cpp -o cpu_vs_cuda `pkg-config --cflags --libs opencv4`


using namespace cv;
using namespace std;

int main()
{
    string videoPath = "video.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened())
    {
        cerr << "Error al abrir el video\n";
        return -1;
    }

    Mat frame, cpuResult;
    cuda::GpuMat gpuFrame, gpuResult;

    const Size ksize(15, 15);

    // ======================
    // CPU
    // ======================
    cap.set(CAP_PROP_POS_FRAMES, 0);
    int frameCount = 0;
    double cpuStart = getTickCount();

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        GaussianBlur(frame, cpuResult, ksize, 0);
        frameCount++;
    }

    double cpuTime = (getTickCount() - cpuStart) / getTickFrequency();
    double cpuFPS = frameCount / cpuTime;

    cout << "CPU total: " << cpuTime * 1000 << " ms\n";
    cout << "CPU FPS:   " << cpuFPS << endl;

    // ======================
    // GPU CUDA
    // ======================
    cap.set(CAP_PROP_POS_FRAMES, 0);
    frameCount = 0;

    // Crear filtro UNA VEZ
    Ptr<cuda::Filter> gauss =
        cuda::createGaussianFilter(
            CV_8UC3,
            CV_8UC3,
            ksize,
            0
        );

    double gpuStart = getTickCount();

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        gpuFrame.upload(frame);
        gauss->apply(gpuFrame, gpuResult);
        gpuResult.download(cpuResult);

        frameCount++;
    }

    double gpuTime = (getTickCount() - gpuStart) / getTickFrequency();
    double gpuFPS = frameCount / gpuTime;

    cout << "GPU total: " << gpuTime * 1000 << " ms\n";
    cout << "GPU FPS:   " << gpuFPS << endl;

    cout << "Speedup GPU vs CPU: " << cpuTime / gpuTime << "x\n";

    return 0;
}

