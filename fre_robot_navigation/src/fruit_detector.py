import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import os
from ultralytics import YOLO
import numpy as np

class FruitDetectorNode(Node):
    def __init__(self):
        super().__init__('fruit_detector_node')
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.detection_publisher = self.create_publisher(
            Image,
            'detected_image',
            10
        )

        try:
            self.model = YOLO('weights/fruit_detector_v1.pt')
            self.model.eval()
            self.get_logger().info("YOLOv11 (Fast) model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Error loading YOLOv11 (Fast) model: {e}")

        self.get_logger().info("Fruit Detector Node Initialized!")

    def image_callback(self, msg):
        CLASS_NAMES = ["apple", "banana", "grape", "lemon", "orange", "strawberry"]
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detected_cv_image = cv_image.copy()

        try:
            results = self.model(rgb_image)

            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                print(f"Detected {len(boxes)} objects")

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
                    cv2.putText(detected_cv_image, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1

                for class_name, count in class_counts.items():
                    print(f"{class_name}: Detected {count} object(s)")

            else:
                print("No objects detected.")

            try:
                detection_msg = self.bridge.cv2_to_imgmsg(detected_cv_image, encoding='bgr8')
                self.detection_publisher.publish(detection_msg)
            except Exception as e:
                self.get_logger().error(f"Could not convert detected image to ROS Image: {e}")

        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()