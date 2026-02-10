#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <chrono>

// ================= Shader helpers =================
GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
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
        std::cout << "Uso: ./cuda_glfw_pbo video.mp4\n";
        return -1;
    }

    // ---------- OpenCV ----------
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
        return -1;

    cv::Mat frame;
    cv::cuda::GpuMat d_frame, d_gray, d_edges;

    auto canny = cv::cuda::createCannyEdgeDetector(50, 150);

    // ---------- GLFW ----------
    glfwInit();
    GLFWwindow* win = glfwCreateWindow(1280, 720, "CUDA PBO Interop", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glewInit();

    GLuint program = createProgram();
    glUseProgram(program);

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // ---------- Texture ----------
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0,
                 GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // ---------- PBO ----------
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLuint pbo;
    glGenBuffers(1, &pbo);
    size_t pboSize = width * height * sizeof(unsigned char);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height, nullptr, GL_DYNAMIC_DRAW);

    // ---------- CUDA ↔ OpenGL ----------
    cudaGraphicsResource* cuda_pbo;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);

    // ---------- Timing ----------
    int frames = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(win))
    {
        cap >> frame;
        if (frame.empty())
            break;

	// ===== CUDA =====
	d_frame.upload(frame);
	cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);
	canny->detect(d_gray, d_edges);

	// ===== DEBUG FIX 2: verificar formato =====
	std::cout << "d_edges: "
          << " type=" << d_edges.type()
          << " step=" << d_edges.step
          << " cols=" << d_edges.cols
          << " rows=" << d_edges.rows
          << std::endl;

	// Map PBO
	uchar* d_pbo = nullptr;
	size_t size = 0;

	cudaGraphicsMapResources(1, &cuda_pbo);
	cudaGraphicsResourceGetMappedPointer(
	    (void**)&d_pbo, &size, cuda_pbo);

	// Copia correcta (2D)
	cudaMemcpy2D(
	    d_pbo,
	    width,
	    d_edges.data,
	    d_edges.step,
	    width,
	    height,
	    cudaMemcpyDeviceToDevice
	);

	cudaGraphicsUnmapResources(1, &cuda_pbo);

        // ===== OpenGL =====
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        width, height,
                        GL_RED, GL_UNSIGNED_BYTE, nullptr);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(win);
        glfwPollEvents();

        frames++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();

    std::cout << "\n===== RESULTADOS (PBO) =====\n";
    std::cout << "Frames: " << frames << "\n";
    std::cout << "FPS promedio: " << frames / secs << "\n";

    cudaGraphicsUnregisterResource(cuda_pbo);
    glfwTerminate();
    return 0;
}

