import imutils
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Getting path of test images
images = Path('test_images').glob('*.png')

# Looping over all images
for imagePath in images:
    # Reading and showing original image
    image = cv2.imread(str(imagePath))
    plt.figure("Original Image")
    plt.imshow(image)

    # Transforming image into greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blurring image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecting edges using Canny algorithm
    edged = cv2.Canny(blurred, 50, 150)

    # Thresholding to obtain a binary image
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Getting all contours of the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Getting the largest contour, which is the digit itself
    largest_contour = max(contours, key=cv2.contourArea)

    # Getting bounding box of contour
    (x, y, w, h) = cv2.boundingRect(largest_contour)

    # Cutting bounding box out of image
    digit = thresh[y:y + h, x:x + w]

    # Some calculations used to divide the image into seven segments
    gW = int(0.33 * w)
    gH = int(0.33 * h)
    lW = int(0.15 * w)
    lH = int(0.15 * h)

    # Some (x,y) coordinates indicating different segments on the image
    segments = [
        ((lW, 0), (w - lW, gH)),  # top
        ((lW, lH), (gW, h // 2)),  # top-left
        ((w - gW, lH), (w - lW, h // 2)),  # top-right
        ((gW, (h // 2) - lH), (w - gW, (h // 2) + lH)),  # center
        ((lW, (h // 2) + lH), (gW, h - lH)),  # bottom-left
        ((w - gW, (h // 2) + lH), (w - lW, h - lH)),  # bottom-right
        ((gW, h - gH), (w - gW, h - lW))  # bottom
    ]

    # Array to hold binary values indicating which segments are on
    segmentsOn = [0] * len(segments)
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # Extracting the segment from the image
        segment = digit[yA:yB, xA:xB]

        # Counting all non-zero pixels
        total = cv2.countNonZero(segment)

        # Calculating area of the segment
        area = (xB - xA) * (yB - yA)

        # If the total of non-zero pixels is greater than 45% of the segment, we mark it as on
        if total / float(area) > 0.45:
            segmentsOn[i] = 1

    # Getting the digit based on which segments are on
    extractedDigit = 0
    if segmentsOn[0] == 0:
        extractedDigit = 6
    elif segmentsOn[6] == 0:
        extractedDigit = 9
    else:
        if segmentsOn[1] == 0:
            if segmentsOn[3] == 0:
                extractedDigit = 1
            else:
                if segmentsOn[5] == 0:
                    extractedDigit = 7
                else:
                    extractedDigit = 3
        else:
            if segmentsOn[2] == 0:
                extractedDigit = 5
            else:
                if segmentsOn[3] == 0:
                    if segmentsOn[4] == 0:
                        extractedDigit = 2
                    else:
                        extractedDigit = 0
                else:
                    extractedDigit = 8

    # Printing the digit, and showing its image
    print("The digit in the image is", extractedDigit)
    plt.figure("Digit")
    plt.imshow(digit, cmap="gray")
    plt.show()
