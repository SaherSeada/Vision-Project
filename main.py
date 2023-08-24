import scipy
import cv2
import matplotlib.pyplot as plt
import numpy as np

# data = scipy.io.loadmat('test_32x32.mat')
# images = data['X']
# labels = data['y']
image = cv2.imread('../cropped/output_images/image_07316_label_2.png')
plt.imshow(image)
plt.show()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 50, 150)

plt.figure(2)
plt.imshow(edged, cmap="gray")
plt.show()

# for number in range(60, 81):
#     image = images[:, :, :, number]
#     plt.figure(number-1)
#     plt.imshow(image)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     plt.figure(number)
#     plt.imshow(edged, cmap="gray")
#     plt.show()

