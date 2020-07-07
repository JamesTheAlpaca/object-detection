import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image



        
im = cv2.imread(r'C:\Users\james\OneDrive\Desktop\logos\gta12345.jpg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()
plt.clf()
plt.close
cv2.imshow('img',output_image)


