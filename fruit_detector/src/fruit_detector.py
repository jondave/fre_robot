#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import traceback

class FruitDetectorNode:
    def __init__(self):
        rospy.init_node('fruit_detector_node', anonymous=True)
        self.bridge = CvBridge()

        # Subscribe to camera image topic
        self.image_sub = rospy.Subscriber('/realsense/color/image_raw', Image, self.image_callback, queue_size=10) # Adjust topic name as needed
        # Publish the detected image
        self.image_pub = rospy.Publisher('detected_image', Image, queue_size=10)

        self.model = None
        try:
            self.model = YOLO("/home/jonathan/catkin_ws/src/fre_robot/fruit_detector/weights/weights_strawberry_v2.pt") # Update with your model path
            self.model.eval()
            rospy.loginfo("YOLOv11 (Fast) model loaded successfully!")
        except Exception as e:
            rospy.logerr(f"Error loading YOLOv11 (Fast) model: {e}")
            traceback.print_exc()

        rospy.loginfo("Fruit Detector Node Initialized!")

    def image_callback(self, msg):
        if self.model is None:
            rospy.logwarn("Model not loaded, skipping inference.")
            return

        CLASS_NAMES = ["strawberry", "other"]  # update as needed

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert ROS Image to CV image: {e}")
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detected_cv_image = cv_image.copy()

        try:
            results = self.model(rgb_image)

            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                rospy.loginfo(f"Detected {len(boxes)} objects")

                class_counts = {}
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    conf = confidences[i]
                    cls_id = class_ids[i]

                    class_name = f"Class {cls_id}"
                    if 0 <= cls_id < len(CLASS_NAMES):
                        class_name = CLASS_NAMES[cls_id]

                    label = f"{class_name}: {conf:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(detected_cv_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(detected_cv_image, label,
                                (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                for class_name, count in class_counts.items():
                    rospy.loginfo(f"{class_name}: Detected {count} object(s)")

            else:
                rospy.loginfo("No objects detected.")

            try:
                detection_msg = self.bridge.cv2_to_imgmsg(detected_cv_image, encoding='bgr8')
                self.image_pub.publish(detection_msg)
            except CvBridgeError as e:
                rospy.logerr(f"Could not convert CV image to ROS Image: {e}")

        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    try:
        node = FruitDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
