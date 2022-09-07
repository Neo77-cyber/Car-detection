import cv2
import numpy as np

import matplotlib.pyplot as plt
import cvlib as cv 
from cvlib.object_detection import draw_bbox
im = cv2. imread("cars-for-sale-parking-sale.jpg")

bbox, label, conf = cv.detect_common_objects( im)
output_image = draw_bbox( im, bbox, label, conf)
plt.imshow( output_image)
plt.show()
print("number of cars in the image is " + str(label.count ("car")))

img = np.zeros((512,512,3), np.uint8)
print(cv2.circle(img,(447,63), 63, (0,0,255), -1))