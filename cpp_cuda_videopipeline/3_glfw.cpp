#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

/*
 * Se compila con: g++ gl_test.cpp -o gl_test -lglfw -lGLEW -lGL
 */


int main() {
    if (!glfwInit()) {
        std::cerr << "Error inicializando GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Test", NULL, NULL);
    if (!window) {
        std::cerr << "No se pudo crear ventana\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Error inicializando GLEW\n";
        return -1;
    }

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

