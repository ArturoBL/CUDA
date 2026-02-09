#include <iostream>
#include <cudnn.h>

int main() {
    std::cout << "cuDNN version: "
              << CUDNN_MAJOR << "."
              << CUDNN_MINOR << "."
              << CUDNN_PATCHLEVEL
              << std::endl;
    return 0;
}
