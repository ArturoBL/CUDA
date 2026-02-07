#include <opencv2/opencv.hpp>
#include <iostream>

// Compilar con: g++ showimage.cpp -o app $(pkg-config --cflags --libs opencv4)

int main(int argc, char** argv) {
    // Check if an image file path was provided as a command line argument
    if (argc != 2) {
        std::cout << "Usage: ./DisplayImage <image_path>" << std::endl;
        return -1;
    }

    // Read the image file specified by the command line argument
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Check for invalid input
    if (image.empty()) {
        std::cout << "Could not read the image: " << argv[1] << std::endl;
        return -1;
    }

    // Create a window for display
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

    // Show the image
    cv::imshow("Display window", image);

    // Wait for a keystroke in the window
    cv::waitKey(0);

    // Close all windows
    cv::destroyAllWindows();

    return 0;
}
