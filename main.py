import cv2
from ObjectDetection import yolo_detector as helper

WEIGHTS = "./yolov3.weights"
CONFIG  = "./yolov3.cfg"
CLASS_FILE = "./coco.names"
    
DnnHelper = helper(WEIGHTS, CONFIG, CLASS_FILE)
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    height, width, channel = img.shape
    
    # Detect Objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (188, 188), (0, 0, 0), True, crop=False)
    outs = DnnHelper.run_net(blob)

    # Show information on the screen
    class_ids, confidences, boxes = DnnHelper.get_data(outs, img.shape)
    DnnHelper.draw_to_screen(img, class_ids, confidences, boxes)

    cv2.imshow("VidCapture", img)

    if cv2.waitKey(1) == 13:
        break