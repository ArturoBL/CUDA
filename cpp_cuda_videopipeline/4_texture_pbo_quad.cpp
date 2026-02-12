#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

/*
 * Se compila con: g++ pbo_test.cpp -o pbo_test -lglfw -lGLEW -lGL
 */


int main() {
    const int width = 800;
    const int height = 600;

    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(width, height, "PBO Test", NULL, NULL);
    glfwMakeContextCurrent(window);

    glewInit();

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 width * height * 3,
                 NULL,
                 GL_DYNAMIC_DRAW);

    // Datos fake (gradiente)
    std::vector<unsigned char> image(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[(y * width + x) * 3 + 0] = x % 256;
            image[(y * width + x) * 3 + 1] = y % 256;
            image[(y * width + x) * 3 + 2] = 0;
        }
    }

    while (!glfwWindowShouldClose(window)) {

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER,
                        0,
                        width * height * 3,
                        image.data());

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D,
                        0, 0, 0,
                        width, height,
                        GL_RGB,
                        GL_UNSIGNED_BYTE,
                        0);

        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

