import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

images = Path('/test_images').glob('*.png')

for image in images:
    plt.figure(number-1)
    plt.imshow(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    largest_contour = max(cnts, key=cv2.contourArea)
    # cnts = imutils.grab_contours(cnts)
    # digitCnts = []
    # loop over the digit area candidates
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(largest_contour)
    # if the contour is sufficiently large, it must be a digit
    # digitCnts.append(c)
    # extract the digit ROI
    roi = thresh[y:y+h, x:x + w]
    dW = int(0.3 * w)
    dH = int(0.3 * h)
    lW = int(0.15 * w)
    lH = int(0.15 * h)
    segments = [
        ((lW, 0), (w, dH)),  # top
        ((lW, 0), (dW, h // 2)),  # top-left
        ((w - dW, dH), (w, h // 2)),  # top-right
        ((dW, (h // 2) - lH), (w-dW, (h // 2) + lH)),  # center
        ((lW, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((lW, h - dH), (w, h))  # bottom
    ]
    on = [0] * len(segments)
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        print(xA, yA, xB, yB)
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.45:
            on[i] = 1
        print(on)
    plt.figure("number")
    plt.imshow(roi, cmap="gray")
    plt.show()


