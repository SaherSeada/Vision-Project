import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

images = Path('test_images').glob('*.png')

for imagePath in images:
    image = cv2.imread(str(imagePath))
    plt.figure("Original Image")
    plt.imshow(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    largest_contour = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(largest_contour)
    roi = thresh[y:y+h, x:x + w]
    gW = int(0.33 * w)
    gH = int(0.33 * h)
    lW = int(0.15 * w)
    lH = int(0.15 * h)
    segments = [
        ((lW, 0), (w - lW, gH)),  # top
        ((lW, lH), (gW, h // 2)),  # top-left
        ((w - gW, lH), (w - lW, h // 2)),  # top-right
        ((gW, (h // 2) - lH), (w - gW, (h // 2) + lH)),  # center
        ((lW, (h // 2) + lH), (gW, h - lH)),  # bottom-left
        ((w - gW, (h // 2) + lH), (w - lW, h - lH)),  # bottom-right
        ((gW, h - gH), (w - gW, h - lW))  # bottom
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
    plt.figure("ROI")
    plt.imshow(roi, cmap="gray")
    plt.show()


