import cv2
from obj_det_wrap import detector
import numpy as np
import os

det = detector('../models/dataset2/train/weights/best.pt')

vid = cv2.VideoCapture('../data/input3.mp4')

roi = np.array([[667,291], [553, 539], [810, 536], [738, 355], [918,365], [877, 306]], np.int32)

intr_count = {}
global_intr_boxes = []

while True:
    
    ret, frame = vid.read()
    if not ret:
        break

    #Skipping frames (remove later)
    for i in range(2):
        ret = vid.grab()
        if not ret:
            break
    

    height, width, _ = frame.shape
    new_h = int(height * 0.5)
    new_w = int(width * 0.5)
    
    frame = cv2.resize(frame,(new_w,new_h))
    
    cv2.polylines(frame, [roi], True, (144, 238, 144), 2)

    bboxes, bdims, conf = det.detect(frame)

    intrusion_boxes = []
    
    for bbox in bboxes:
        (x1, y1, x2, y2) = bbox
        #(x, y, w, h) = dims
        
        cx = int((x1 + x2)/ 2)
        cy = int((y1 + y2)/ 2)
        
        #cv2.circle(frame, (cx, cy), 0, (255, 0 , 0), -1)

        res = cv2.pointPolygonTest(np.array(roi, np.int32), (int(cx), int(cy)), False)
        if res>=0:
            intrusion_boxes.append(bbox)
            global_intr_boxes.append(bbox)
    
    folder_name = 'images'
    
    for bbox in intrusion_boxes:
        (x1, y1, x2, y2) = bbox
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0, 0, 255),5)
        
        cv2.putText(frame, 'Intruder Detected', (10, 30), 0, 0.8, (0, 0, 255), 2)

        # if intr_count>0:
        #     cv2.rectangle(frame, (x1,y1),(x2,y2),(0, 0, 255),5)
        #     cv2.putText(frame, 'Intruder Detected', (10, 30), 0, 0.8, (0, 0, 255), 2)
        #     cv2.putText(frame, 'Alarm Raised', (10, 55), 0, 0.8, (0, 0, 255), 2)

    cv2.imshow('frame',frame)
    key = cv2.waitKey(33)
    if key == 27:
        break
img_num = 0
for bbox in global_intr_boxes:
    (x1, y1, x2, y2) = bbox
    image_filename = f"Detection case_{img_num}.png"
    cv2.imwrite(os.path.join(folder_name, image_filename), frame[y1:y2, x1:x2])
    img_num+=1

global_intr_boxes.clear()
cv2.destroyAllWindows()