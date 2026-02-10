#include <iostream>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

/*Compilar con:
 *
 g++ main.cpp -o canny_gl   `pkg-config --cflags --libs opencv4`   -lglfw -lGLEW -lGL   -lcuda -lcudart -I/usr/local/cuda/include   -L/usr/local/cuda/lib64

 Ejecutar con:
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./canny_gl
*/

static const char* vs_src = R"(
#version 330 core
out vec2 uv;
void main() {
    vec2 pos = vec2((gl_VertexID & 1) << 1, (gl_VertexID & 2));
    uv = pos * 0.5;
    gl_Position = vec4(pos - 1.0, 0.0, 1.0);
}
)";

static const char* fs_src = R"(
#version 330 core
in vec2 uv;
out vec4 frag;
uniform sampler2D tex;
void main() {
    float v = texture(tex, uv).r;
    frag = vec4(v, v, v, 1.0);
}
)";

GLuint compile(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
}

GLuint makeProgram()
{
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs_src));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs_src));
    glLinkProgram(p);
    return p;
}

int main()
{
    // ------------------------------------------------------------
    // GLFW / OpenGL
    // ------------------------------------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    GLFWwindow* win = glfwCreateWindow(1280, 720, "CUDA Canny Video", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glewInit();

    // ------------------------------------------------------------
    // CUDA context AFTER OpenGL
    // ------------------------------------------------------------
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    // ------------------------------------------------------------
    // Video file
    // ------------------------------------------------------------
    cv::VideoCapture cap("video.mp4", cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir video.mp4\n";
        return -1;
    }

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::Mat frame, frame_gray;
    cv::cuda::GpuMat d_gray, d_edges;

    auto canny = cv::cuda::createCannyEdgeDetector(80, 150);

    // ------------------------------------------------------------
    // OpenGL texture + PBO
    // ------------------------------------------------------------
    GLuint tex, pbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED,
                 width, height, 0,
                 GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 width * height, nullptr, GL_STREAM_DRAW);

    // ------------------------------------------------------------
    // CUDA–OpenGL interop
    // ------------------------------------------------------------
    cudaGraphicsResource* cuda_pbo;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // ------------------------------------------------------------
    // Shader
    // ------------------------------------------------------------
    GLuint prog = makeProgram();
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);

    // ------------------------------------------------------------
    // FPS
    // ------------------------------------------------------------
    int frames = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------
    // Loop
    // ------------------------------------------------------------
    while (!glfwWindowShouldClose(win))
    {
        cap >> frame;

        // fin del video → reiniciar
        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        d_gray.upload(frame_gray);

        canny->detect(d_gray, d_edges);

        // --- CUDA → PBO
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_pbo));

        uchar* d_pbo = nullptr;
        size_t size = 0;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            (void**)&d_pbo, &size, cuda_pbo));

        CUDA_CHECK(cudaMemcpy2DAsync(
            d_pbo,
            width,
            d_edges.ptr<uchar>(),
            d_edges.step,
            width,
            height,
            cudaMemcpyDeviceToDevice
        ));

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo));

        // --- OpenGL draw
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                         width, height,
                         GL_RED, GL_UNSIGNED_BYTE, nullptr);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(win);
        glfwPollEvents();

        // FPS
        frames++;
        auto t1 = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(t1 - t0).count() >= 1.0) {
            std::cout << "FPS: " << frames << std::endl;
            frames = 0;
            t0 = t1;
        }
    }

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    cudaGraphicsUnregisterResource(cuda_pbo);
    glfwTerminate();
    return 0;
}

