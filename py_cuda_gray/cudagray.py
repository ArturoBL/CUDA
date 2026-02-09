import cv2
import numpy as np

img = cv2.imread("foto.jpg")
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)

gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

gray = gpu_gray.download()
cv2.imshow("Gray CUDA", gray)
cv2.waitKey(0)
