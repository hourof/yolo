import torch
import cv2
import mss
import numpy
#MODEL
model = torch.hub.load("./", "yolov5n", source='local')

monitor = {"top": 0, "left": 0, "width": 800, "height": 700}

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(monitor)
        srcimg = numpy.array(screenshot)
        # Inference
        result = model(srcimg)

        # Result
        result.show()
    #Img
    img = cv2.imread('data/images/bus.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

