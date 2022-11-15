import numpy as np
import cv2

class yolo_detector:
    def __init__(self, weights_path, config_path, class_path):
        """
        It reads the weights and config files, and then reads the class names from the class file.
        
        :param weights_path: The path to the weights file
        :param config_path: Path to the yolov3 config file
        :param class_path: path to the text file containing class names seperated by new line
        """
        self.config = {
            "colors": {
                "bb_color":    (255, 0, 0),
                "text_color":  (255, 255, 255)
            },
            "conf": {
                "conf_threshold": 0.6,
                "nms_threshold": 0.5
            }
        }
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(class_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
    def run_net(self, input):
        """
        Takes the output of the YOLOv3 network and returns the bounding boxes, confidence scores, and
        class IDs.
        
        :param net: The network object
        :param input: The input image
        """
        layer_name = self.net.getLayerNames()
        output_layer = [layer_name[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.net.setInput(input)
        return self.net.forward(output_layer)

    def get_data(self, outs, shape):
        """
        It takes the output of the YOLO model, and returns a dictionary containing the class IDs,
        confidences, and bounding boxes of the detected objects
        
        :param outs: The output of the YOLO model
        :param shape: The shape of the image
        :return: A tuple with three values: class_ids, confidences, boxes.
        """
        height, width, _ = shape

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    # Object detection
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Reactangle Cordinate
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return (class_ids, confidences, boxes)
    
    def draw_to_screen(self, img, class_ids, confidences, boxes):
        """
        Takes the image, the class_ids, the confidences, and the boxes and draws the bounding boxes to
        the image
        
        :param img: The image to draw the bounding boxes on
        :param class_ids: The class ID of the detected object
        :param confidences: The confidence of the detection
        :param boxes: the bounding boxes of the detected objects
        """
        conf_cfg = self.config["conf"]
        color_cfg = self.config["colors"]

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_cfg["conf_threshold"], conf_cfg["nms_threshold"])
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]]) + str(round(confidences[i], 2))

                cv2.rectangle(img, (x, y), (x + w, y + h), color_cfg["bb_color"], 2)
                cv2.putText(img, label, (x, y - 30), font, 2, color_cfg["text_color"], 3)