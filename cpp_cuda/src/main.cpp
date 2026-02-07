#include <iostream>

extern "C" void launch_kernel();

int main() {
    std::cout << "Lanzando kernel desde C++..." << std::endl;
    launch_kernel();
    std::cout << "Listo ✅" << std::endl;
    return 0;
}
