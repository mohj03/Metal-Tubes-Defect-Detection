import numpy as np
import pandas as pd
import cv2 as cv    
import skimage as ski
import matplotlib.pyplot as plt

img = cv.imread("TubesDataset/0008.png", cv.IMREAD_GRAYSCALE)

blurred = cv.GaussianBlur(img, (13, 13), 0)
canny = cv.Canny(blurred, 30, 100)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel, iterations=5)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original")

ax[1].imshow(canny, cmap='gray')
ax[1].set_title("Canny")

ax[2].imshow(closed, cmap='gray')
ax[2].set_title("Morphological Closing")

plt.tight_layout()
plt.show()

