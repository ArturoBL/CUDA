#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

/*
 * Se compila con:
 *
g++ cuda_opengl_streams.cpp -o app \
`pkg-config --cflags --libs opencv4` \
-I/usr/local/cuda/include \
-L/usr/local/cuda/lib64 \
-lglfw -lGLEW -lGL -lcudart

*/


using namespace std;

// ================= SHADERS =================

const char* vertexShaderSrc = R"(#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSrc = R"(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;

void main()
{
    FragColor = texture(screenTexture, TexCoord);
}
)";

GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char info[512];
        glGetShaderInfoLog(shader, 512, NULL, info);
        cout << "Shader compile error:\n" << info << endl;
    }
    return shader;
}

// ===================== MAIN =====================

int main()
{
    std::string video_path = "video.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cout << "No se pudo abrir la cámara\n";
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // ---------- GLFW ----------
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Streams", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // sin vsync

    glewExperimental = GL_TRUE;
    glewInit();

    glViewport(0, 0, width, height);

    // ---------- SHADERS ----------
    GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertex);
    glAttachShader(shaderProgram, fragment);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    // ---------- QUAD ----------
    float vertices[] = {
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f,

        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 0.0f
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // ---------- TEXTURE ----------
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 width, height,
                 0, GL_BGR,
                 GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // ---------- PBO ----------
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 width * height * 3,
                 NULL,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsResource* cuda_pbo;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);

    // ---------- CUDA ----------
    cv::cuda::GpuMat d_frame, d_blur;
    auto gaussian = cv::cuda::createGaussianFilter(
        CV_8UC3, CV_8UC3, cv::Size(15,15), 3);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    cv::Mat frame;
    cv::TickMeter tm;

    while (!glfwWindowShouldClose(window))
    {
        cap >> frame;
        if (frame.empty()) break;

        tm.start();

        // Upload async
        d_frame.upload(frame, cvStream);

        // Blur async
        gaussian->apply(d_frame, d_blur, cvStream);

        // Map PBO
        cudaGraphicsMapResources(1, &cuda_pbo, stream);

        uchar3* dptr;
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&dptr,
                                             &num_bytes,
                                             cuda_pbo);

        // Async copy GPU→GPU
        cudaMemcpy2DAsync(
            dptr,
            width * 3,
            d_blur.ptr(),
            d_blur.step,
            width * 3,
            height,
            cudaMemcpyDeviceToDevice,
            stream
        );

        // IMPORTANT: sincronizar antes de unmap
        cudaStreamSynchronize(stream);

        cudaGraphicsUnmapResources(1, &cuda_pbo, stream);

        // Update texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexSubImage2D(GL_TEXTURE_2D,
                        0, 0, 0,
                        width, height,
                        GL_BGR,
                        GL_UNSIGNED_BYTE,
                        0);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();

        tm.stop();
        cout << "FPS: " << 1000.0 / tm.getTimeMilli() << endl;
        tm.reset();
    }

    cudaStreamDestroy(stream);
    cudaGraphicsUnregisterResource(cuda_pbo);

    glfwTerminate();
    return 0;
}

