#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>

/* 
 * Se compila con: g++ cuda_opengl_pipeline.cpp -o cuda_gl \
`pkg-config --cflags --libs opencv4` \
-lglfw -lGLEW -lGL -lcuda -lcudart

 O bien:
 nvcc 5_interop_cuda_pbo.cpp -o 5_interop_cuda_pbo `pkg-config --cflags --libs opencv4` -lglfw -lGLEW -lGL

*/

int main() {

    std::string video_path = "video.mp4";
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el video\n";
        return -1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Resolución: " << width << " x " << height << std::endl;

    // ---- Inicializar GLFW ----
    if (!glfwInit()) {
        std::cerr << "Error inicializando GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA + OpenGL", NULL, NULL);
    if (!window) {
        std::cerr << "No se pudo crear ventana\n";
        glfwTerminate();
        return -1;
    }
    glfwSwapInterval(0);
    glfwMakeContextCurrent(window);
    glewInit();

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;

    // ---- Inicializar CUDA ----
    cudaSetDevice(0);

    // ---- Crear textura OpenGL ----
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // ---- Crear PBO ----
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 width * height * 3,
                 NULL,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ---- Registrar PBO con CUDA ----
    cudaGraphicsResource* cuda_pbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource,
                                 pbo,
                                 cudaGraphicsMapFlagsWriteDiscard);

    // ---- OpenCV CUDA setup ----
    cv::cuda::GpuMat d_frame, d_blur;
    cv::Mat frame;

    auto gaussian = cv::cuda::createGaussianFilter(
        CV_8UC3,
        CV_8UC3,
        cv::Size(15, 15),
        3
    );

    cv::TickMeter tm;
    int frame_count = 0;

    while (!glfwWindowShouldClose(window)) {

        if (!cap.read(frame))
            break;

        tm.start();

        // Upload CPU → GPU
        d_frame.upload(frame);

        // Gaussian blur en GPU
        gaussian->apply(d_frame, d_blur);

        // ---- CUDA → PBO ----
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);

        uchar* d_pbo_ptr;
        size_t num_bytes;

        cudaGraphicsResourceGetMappedPointer(
            (void**)&d_pbo_ptr,
            &num_bytes,
            cuda_pbo_resource
        );

	cudaMemcpy2D(
	    d_pbo_ptr,                 // destino
	    width * 3,                 // pitch destino (PBO es compacto)
	    d_blur.ptr(),              // origen
	    d_blur.step,               // pitch real del GpuMat
	    width * 3,                 // ancho real en bytes
	    height,                    // número de filas
	    cudaMemcpyDeviceToDevice
	);


        cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

        // ---- Actualizar textura ----
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexSubImage2D(GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        width,
                        height,
                        GL_BGR,
                        GL_UNSIGNED_BYTE,
                        0);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // ---- Render ----
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
	glTexCoord2f(0, 1); glVertex2f(-1, -1);
	glTexCoord2f(1, 1); glVertex2f( 1, -1);
	glTexCoord2f(1, 0); glVertex2f( 1,  1);
	glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();

        tm.stop();
        frame_count++;
    }

    double fps = frame_count / tm.getTimeSec();
    std::cout << "FPS promedio: " << fps << std::endl;

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glfwTerminate();

    return 0;
}

