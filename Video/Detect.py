import numpy as np
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path = r'C:\Users\SEMMA\Desktop\yolov5-master\Video\model\carros.pt')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    detect = model(frame)

    cv2.imshow('Detector de valvulas', np.squeeze(detect.render()))

    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
