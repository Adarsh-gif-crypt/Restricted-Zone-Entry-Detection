import cv2
from obj_det_wrap import detector
import numpy as np

od = detector('../models/dataset2/train/weights/best.pt')

img = cv2.imread('../data/framecap2.png')

h, w, _ = img.shape

new_h = int(h * 0.5)
new_w = int(w * 0.5)

resize = cv2.resize(img,(new_w,new_h))

bboxes, dims, conf = od.detect(resize)
print('Box in this frame:',bboxes)

for bbox in bboxes:
    (x1,y1,x2,y2) = bbox
    cv2.rectangle(resize, (x1,y1),(x2,y2),(0, 0, 255),5)

cv2.imshow('Img',resize)

cv2.waitKey(0)
cv2.destroyAllWindows()