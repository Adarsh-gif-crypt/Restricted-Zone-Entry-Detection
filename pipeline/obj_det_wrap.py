from ultralytics import YOLO
import numpy as np


class detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):

        h, w, c = img.shape

        results = self.model.predict(source = img.copy(), save = False, save_txt = False)
        result = results[0]

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype = 'int')
        bdims = np.array(result.boxes.xywh.cpu(), dtype = 'int')
        conf = np.array(result.boxes.conf.cpu(), dtype = 'int')

        return bboxes, bdims, conf

