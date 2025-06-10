import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import torch
import os
from ultralytics import YOLO
import numpy as np
import csv
import time
import pandas as pd # Added for easier CSV handling
from sklearn.cluster import DBSCAN # Added for DBSCAN clustering

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point
import geometry_msgs.msg

class FruitDetectorNode(Node):
    def __init__(self):
        super().__init__('fruit_detector_node')
        self.bridge = CvBridge()

        # Subscriptions
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.detection_publisher = self.create_publisher(
            Image,
            'detected_image',
            10
        )

        # YOLO Model Loading
        try:
            self.model = YOLO('weights/fruit_detector_v1.pt')
            self.model.eval()
            self.get_logger().info("YOLOv11 (Fast) model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Error loading YOLOv11 (Fast) model: {e}")

        # TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera Intrinsics
        self.camera_intrinsics = None
        self.get_logger().info("Waiting for camera info...")

        # CSV File Setup
        self.csv_file_path = os.path.join(os.path.expanduser('~'), 'fruit_detections.csv')
        self.clustered_csv_file_path = os.path.join(os.path.expanduser('~'), 'clustered_fruit_detections.csv')
        self.csv_header = ['fruit_type', 'map_x', 'map_y']
        self.clustered_csv_header = ['fruit_type', 'cluster_id', 'map_x', 'map_y']
        self.initialize_csv()

        # Data storage for clustering
        self.detected_fruit_points = [] # To store all detected points before clustering

        # Clustering Timer
        self.clustering_timer = self.create_timer(180.0, self.timer_callback) # 180 seconds = 3 minutes
        self.get_logger().info("Clustering timer set for 3 minutes.")

        self.get_logger().info("Fruit Detector Node Initialized!")

    def initialize_csv(self):
        """Initializes the CSV files with headers if they don't exist."""
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.csv_header)
            self.get_logger().info(f"Created new raw detections CSV file: {self.csv_file_path}")
        else:
            self.get_logger().info(f"Appending to existing raw detections CSV file: {self.csv_file_path}")
        
        if not os.path.exists(self.clustered_csv_file_path):
            with open(self.clustered_csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.clustered_csv_header)
            self.get_logger().info(f"Created new clustered detections CSV file: {self.clustered_csv_file_path}")
        else:
            self.get_logger().info(f"Appending to existing clustered detections CSV file: {self.clustered_csv_file_path}")


    def camera_info_callback(self, msg):
        """Stores camera intrinsic parameters."""
        self.camera_intrinsics = msg.k
        self.get_logger().info("Received camera info.")
        self.destroy_subscription(self.camera_info_subscription)

    def image_callback(self, msg):
        if self.camera_intrinsics is None:
            self.get_logger().warn("Skipping image callback: Camera intrinsics not received yet.")
            return

        CLASS_NAMES = ["Apple", "Banana", "Grapes", "Lemon", "Orange"]

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detected_cv_image = cv_image.copy()
        
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not transform '{msg.header.frame_id}' to 'map': {ex}")
            return

        try:
            results = self.model(rgb_image)

            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                self.get_logger().info(f"Detected {len(boxes)} objects")

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

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    assumed_depth = 1.0 # meters - ADJUST THIS IF YOU HAVE A BETTER ESTIMATE OR DEPTH SENSOR
                    
                    fx = self.camera_intrinsics[0]
                    fy = self.camera_intrinsics[4]
                    cx = self.camera_intrinsics[2]
                    cy = self.camera_intrinsics[5]

                    point_camera_x = (center_x - cx) * assumed_depth / fx
                    point_camera_y = (center_y - cy) * assumed_depth / fy
                    point_camera_z = assumed_depth

                    point_in_camera_frame = geometry_msgs.msg.PointStamped()
                    point_in_camera_frame.header.frame_id = msg.header.frame_id
                    point_in_camera_frame.header.stamp = msg.header.stamp
                    point_in_camera_frame.point.x = point_camera_x
                    point_in_camera_frame.point.y = point_camera_y
                    point_in_camera_frame.point.z = point_camera_z

                    try:
                        point_in_map_frame = do_transform_point(point_in_camera_frame, transform)
                        map_x = point_in_map_frame.point.x
                        map_y = point_in_map_frame.point.y
                        
                        self.get_logger().info(f"Detected {class_name} at map coordinates: ({map_x:.2f}, {map_y:.2f})")
                        
                        # Store for clustering later
                        self.detected_fruit_points.append({'fruit_type': class_name, 'map_x': map_x, 'map_y': map_y})

                        # Save to raw detections CSV immediately
                        with open(self.csv_file_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([class_name, map_x, map_y])

                    except TransformException as ex:
                        self.get_logger().error(f"Could not transform point from '{msg.header.frame_id}' to 'map': {ex}")
                        
            else:
                self.get_logger().info("No objects detected.")

            try:
                detection_msg = self.bridge.cv2_to_imgmsg(detected_cv_image, encoding='bgr8')
                self.detection_publisher.publish(detection_msg)
            except Exception as e:
                self.get_logger().error(f"Could not convert detected image to ROS Image: {e}")

        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")

    def timer_callback(self):
        """Callback for the clustering timer, triggers the clustering process."""
        self.get_logger().info("3 minutes elapsed. Initiating clustering process...")
        self.perform_clustering()
        # Optionally, you can reset the timer or stop it if clustering is a one-time event
        # self.clustering_timer.cancel() # Uncomment to stop the timer after one run

    def perform_clustering(self):
        """Reads data from CSV, performs DBSCAN clustering, and saves results."""
        if not os.path.exists(self.csv_file_path) or os.stat(self.csv_file_path).st_size == 0:
            self.get_logger().warn("No fruit detection data found in CSV for clustering.")
            return

        try:
            # Read all detected points from the CSV
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                self.get_logger().warn("CSV file is empty, no data to cluster.")
                return

            clustered_data = []

            # Iterate through each unique fruit type for clustering
            for fruit_type in df['fruit_type'].unique():
                fruit_df = df[df['fruit_type'] == fruit_type]
                # Keep original index to map back to original data if needed
                points_with_original_index = fruit_df[['map_x', 'map_y']].reset_index()
                points = points_with_original_index[['map_x', 'map_y']].values

                if len(points) < 1: # DBSCAN needs at least 1 point, but 2 for a cluster beyond noise
                    self.get_logger().info(f"Not enough points to cluster {fruit_type}. Skipping.")
                    continue
                elif len(points) == 1: # Handle single isolated point as a cluster of size 1
                    clustered_data.append({
                        'fruit_type': fruit_type,
                        'cluster_id': 0, # Assign a default cluster ID
                        'map_x': points[0][0],
                        'map_y': points[0][1]
                    })
                    continue

                # DBSCAN parameters (tune these!)
                # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
                db = DBSCAN(eps=0.1, min_samples=1).fit(points) # Adjusted min_samples for flexibility
                labels = db.labels_

                unique_labels = set(labels)
                for k in unique_labels:
                    if k == -1: # Noise points - you might want to include these or discard them
                        self.get_logger().info(f"Skipping noise points for {fruit_type}.")
                        continue
                    
                    class_member_mask = (labels == k)
                    cluster_points_raw = points[class_member_mask] # These are numpy arrays of [x, y]
                    
                    self.get_logger().info(f"Found cluster {k} for {fruit_type} with {len(cluster_points_raw)} points.")

                    # Limit to a maximum of 5 points from this cluster
                    points_to_add = min(len(cluster_points_raw), 5) # Take up to 5 points

                    for i in range(points_to_add):
                        x = cluster_points_raw[i][0]
                        y = cluster_points_raw[i][1]
                        clustered_data.append({
                            'fruit_type': fruit_type,
                            'cluster_id': k,
                            'map_x': x,
                            'map_y': y
                        })
                        self.get_logger().info(f"Adding point from cluster {k} for {fruit_type}: ({x:.2f}, {y:.2f})")
            
            # Save the clustered data to a new CSV
            if clustered_data:
                # Use 'w' mode to overwrite previous clustered data if the timer runs multiple times
                # If you want to append across multiple clustering runs, change 'w' to 'a' and handle headers carefully
                with open(self.clustered_csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.clustered_csv_header) # Write header only once
                    for row_data in clustered_data:
                        writer.writerow([row_data['fruit_type'], row_data['cluster_id'], row_data['map_x'], row_data['map_y']])
                self.get_logger().info(f"Clustered fruit detections saved to: {self.clustered_csv_file_path}")
            else:
                self.get_logger().warn("No clusters formed or no data to save after filtering.")

        except Exception as e:
            self.get_logger().error(f"Error during clustering process: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()