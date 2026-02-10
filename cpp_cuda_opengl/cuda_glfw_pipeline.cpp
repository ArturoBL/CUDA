#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>

// ================= Shader helpers =================
GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    return shader;
}

GLuint createProgram()
{
    const char* vs = R"(#version 330 core
        out vec2 uv;
        void main() {
            vec2 pos[4] = vec2[](
                vec2(-1,-1), vec2(1,-1),
                vec2(-1,1),  vec2(1,1)
            );
            vec2 tex[4] = vec2[](
                vec2(0,0), vec2(1,0),
                vec2(0,1), vec2(1,1)
            );
            gl_Position = vec4(pos[gl_VertexID],0,1);
            uv = tex[gl_VertexID];
        })";

    const char* fs = R"(#version 330 core
        in vec2 uv;
        out vec4 color;
        uniform sampler2D tex;
        void main() {
            float v = texture(tex, uv).r;
            color = vec4(v, v, v, 1.0);
        })";

    GLuint p = glCreateProgram();
    glAttachShader(p, compileShader(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compileShader(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    return p;
}

// ==================================================

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Uso: ./cuda_glfw_pipeline video.mp4\n";
        return -1;
    }

    // ---------- OpenCV ----------
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cerr << "No se pudo abrir el video\n";
        return -1;
    }

    cv::Mat frame, edges_cpu;
    cv::cuda::GpuMat d_frame, d_gray, d_edges;

    auto canny = cv::cuda::createCannyEdgeDetector(50, 150);
    cv::cuda::Stream stream;

    // ---------- GLFW ----------
    if (!glfwInit())
        return -1;

    GLFWwindow* win = glfwCreateWindow(1280, 720, "CUDA + OpenGL", nullptr, nullptr);
    glfwMakeContextCurrent(win);

    glewInit();

    GLuint program = createProgram();
    glUseProgram(program);

    // ---------- OpenGL texture ----------
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // ---------- Timing ----------
    int frames = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(win))
    {
        cap >> frame;
        if (frame.empty())
            break;

        // ===== CUDA processing =====
        d_frame.upload(frame, stream);
        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY, 0, stream);
        canny->detect(d_gray, d_edges, stream);
        stream.waitForCompletion();

        // Download SOLO para este ejemplo
        d_edges.download(edges_cpu);

        // ===== OpenGL upload =====
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            edges_cpu.cols,
            edges_cpu.rows,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            edges_cpu.data
        );

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(win);
        glfwPollEvents();

        frames++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();

    std::cout << "\n===== RESULTADOS =====\n";
    std::cout << "Frames: " << frames << "\n";
    std::cout << "FPS promedio: " << frames / secs << "\n";

    glfwTerminate();
    return 0;
}

