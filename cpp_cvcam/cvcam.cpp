#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Try opening the default camera (index 0) with the V4L2 backend
    cv::VideoCapture cap(0, cv::CAP_V4L2); 

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera with CAP_V4L2 backend." << std::endl;
        // Optionally, try opening without specifying a backend
        cap.open(0);
        if (!cap.isOpened()) {
             std::cerr << "Error: Could not open camera with default backend either." << std::endl;
             return -1;
        }
        std::cerr << "Warning: Opened with default backend instead." << std::endl;
    }

    // Set properties, e.g., resolution and codec for better performance (MJPG is common)
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(cv::CAP_PROP_FPS, 187.0);

    // Loop to read frames and display them
    cv::Mat frame;
    while (true) {
        cap >> frame; // Read a new frame from the camera

        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            break;
        }

        // Display the frame
        cv::imshow("V4L2 Camera Feed", frame);

        // Break the loop if the 'q' key is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera and destroy windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

