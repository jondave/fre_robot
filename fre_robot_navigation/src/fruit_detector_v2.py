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
import pandas as pd
from sklearn.cluster import DBSCAN

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

        # --- Image Saving Setup ---
        self.image_save_directory = os.path.join(os.path.expanduser('~'), 'fruit_detection_images')
        os.makedirs(self.image_save_directory, exist_ok=True)
        self.get_logger().info(f"Saving detected images to: {self.image_save_directory}")

        # CSV File Setup
        self.csv_file_path = os.path.join(os.path.expanduser('~'), 'fruit_detections.csv')
        # Add a new header for the image file path
        self.csv_header = ['timestamp', 'fruit_type', 'map_x', 'map_y', 'image_filename']
        
        self.clustered_csv_file_path = os.path.join(os.path.expanduser('~'), 'clustered_fruit_detections.csv')
        self.clustered_csv_header = ['fruit_type', 'cluster_id', 'map_x', 'map_y', 'representative_image']
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
        detected_cv_image = cv_image.copy() # Make a copy to draw on

        current_timestamp = time.time()
        
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
                    color = (0, 255, 0) # Green color for bounding box
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
                        
                        # --- Save image for this detection ---
                        # Create a unique filename for the detected image
                        image_filename = f"{int(current_timestamp * 1000)}_{class_name}_{i}.png"
                        image_filepath = os.path.join(self.image_save_directory, image_filename)
                        
                        # Crop the bounding box area and save it
                        # Ensure bounding box coordinates are within image bounds
                        x1_clip = max(0, x1)
                        y1_clip = max(0, y1)
                        x2_clip = min(cv_image.shape[1], x2)
                        y2_clip = min(cv_image.shape[0], y2)
                        
                        cropped_image = cv_image[y1_clip:y2_clip, x1_clip:x2_clip]
                        
                        if cropped_image.size > 0: # Ensure the cropped image is not empty
                            cv2.imwrite(image_filepath, cropped_image)
                            self.get_logger().info(f"Saved cropped detection image: {image_filepath}")
                        else:
                            self.get_logger().warn(f"Cropped image for {class_name} at ({x1},{y1})-({x2},{y2}) was empty.")
                            image_filename = "" # Don't record a filename if image wasn't saved

                        # Store for clustering later (including image_filename)
                        self.detected_fruit_points.append({
                            'fruit_type': class_name,
                            'map_x': map_x,
                            'map_y': map_y,
                            'image_filename': image_filename # Store the filename
                        })

                        # Save to raw detections CSV immediately
                        with open(self.csv_file_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([current_timestamp, class_name, map_x, map_y, image_filename])

                    except TransformException as ex:
                        self.get_logger().error(f"Could not transform point from '{msg.header.frame_id}' to 'map': {ex}")
                        
            else:
                self.get_logger().info("No objects detected.")

            # Publish the image with bounding boxes drawn on it
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
        # If you want it to run continuously, remove the cancel() and consider clearing self.detected_fruit_points
        self.detected_fruit_points = [] # Clear points after clustering to avoid re-clustering old data

    def perform_clustering(self):
        """Reads data from `self.detected_fruit_points`, performs DBSCAN clustering, and saves results."""
        if not self.detected_fruit_points:
            self.get_logger().warn("No fruit detection data collected for clustering.")
            return

        try:
            df = pd.DataFrame(self.detected_fruit_points)
            if df.empty:
                self.get_logger().warn("DataFrame is empty, no data to cluster.")
                return

            clustered_data_to_save = []

            # Iterate through each unique fruit type for clustering
            for fruit_type in df['fruit_type'].unique():
                fruit_df = df[df['fruit_type'] == fruit_type].copy() # Make a copy to avoid SettingWithCopyWarning
                points = fruit_df[['map_x', 'map_y']].values

                if len(points) < 1:
                    self.get_logger().info(f"Not enough points to cluster {fruit_type}. Skipping.")
                    continue
                elif len(points) == 1:
                    # For a single point, we consider it a cluster of size 1
                    # Use .iloc[0] to get the first (and only) row
                    single_point_data = fruit_df.iloc[0] 
                    clustered_data_to_save.append({
                        'fruit_type': fruit_type,
                        'cluster_id': 0, # Assign a default cluster ID for singletons
                        'map_x': single_point_data['map_x'],
                        'map_y': single_point_data['map_y'],
                        'representative_image': single_point_data['image_filename'] # Use its own image
                    })
                    continue

                # DBSCAN parameters (tune these!)
                # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
                db = DBSCAN(eps=0.1, min_samples=1).fit(points) # Adjusted min_samples for flexibility
                labels = db.labels_
                fruit_df['cluster_id'] = labels # Add cluster IDs to the DataFrame

                unique_labels = set(labels)
                for k in unique_labels:
                    if k == -1: # Noise points - you might want to include these or discard them
                        self.get_logger().info(f"Skipping noise points for {fruit_type}.")
                        continue
                    
                    cluster_df = fruit_df[fruit_df['cluster_id'] == k]
                    
                    self.get_logger().info(f"Found cluster {k} for {fruit_type} with {len(cluster_df)} points.")

                    # Limit to a maximum of 5 points from this cluster for saving, if desired
                    # The prompt asks to save "the image with the bounding box to the csv file",
                    # which implies one image per entry. For a cluster, we need a representative image.
                    # We'll pick one, e.g., the first valid image, or the one closest to the centroid.
                    
                    if not cluster_df.empty:
                        # Option: Choose a representative image for the cluster
                        # For simplicity, let's take the image from the first point in the cluster
                        representative_image_filename = ""
                        # Find the first non-empty image filename in the cluster
                        for filename in cluster_df['image_filename']:
                            if filename: # Check if filename is not empty
                                representative_image_filename = filename
                                break
                        
                        # Calculate the centroid of the cluster
                        centroid_x = cluster_df['map_x'].mean()
                        centroid_y = cluster_df['map_y'].mean()

                        clustered_data_to_save.append({
                            'fruit_type': fruit_type,
                            'cluster_id': k,
                            'map_x': centroid_x, # Save centroid for cluster
                            'map_y': centroid_y, # Save centroid for cluster
                            'representative_image': representative_image_filename
                        })
                        self.get_logger().info(f"Added cluster {k} for {fruit_type} at centroid: ({centroid_x:.2f}, {centroid_y:.2f}) with image: {representative_image_filename}")
            
            # Save the clustered data to a new CSV
            if clustered_data_to_save:
                # Use 'w' mode to overwrite previous clustered data if the timer runs multiple times
                with open(self.clustered_csv_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.clustered_csv_header) # Write header only once
                    for row_data in clustered_data_to_save:
                        writer.writerow([row_data['fruit_type'], row_data['cluster_id'], row_data['map_x'], row_data['map_y'], row_data['representative_image']])
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