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
from sklearn.cluster import DBSCAN # Used for grouping nearby fruit detections
from scipy.spatial.distance import euclidean
import tf2_ros

# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk
import threading
import queue

# KNOWN_TREE_LOCATIONS and PROXIMITY_THRESHOLD are now primarily for navigation
# and understanding spatial context, not for filtering the final output CSV.
# You can keep them if your navigation relies on them.
# For the purpose of the final CSV output, we will ignore them as per new rule.

# --- GUI Class (unchanged from previous response) ---
class FruitDetectionGUI:
    def __init__(self, master, message_queue):
        self.master = master
        self.master.title("Fruit Detection Dashboard")
        self.master.geometry("400x200") # Set initial window size

        self.message_queue = message_queue

        self.fruit_label = ttk.Label(self.master, text="Waiting for detections...", font=("Arial", 16))
        self.fruit_label.pack(pady=20)

        self.last_detected_fruit_label = ttk.Label(self.master, text="", font=("Arial", 12))
        self.last_detected_fruit_label.pack(pady=5)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close event

        self.master.after(100, self.check_queue) # Start periodically checking the queue

    def check_queue(self):
        try:
            while True:
                fruit_info = self.message_queue.get_nowait()
                self.update_gui(fruit_info)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.check_queue)

    def update_gui(self, fruit_info):
        fruit_type = fruit_info['fruit_type']
        map_x = fruit_info['map_x']
        map_y = fruit_info['map_y']
        
        self.fruit_label.config(text=f"Detected: {fruit_type}")
        self.last_detected_fruit_label.config(text=f"At: X={map_x:.2f}, Y={map_y:.2f}")

    def on_closing(self):
        self.master.quit()
        self.master.destroy()

# --- FruitDetectorNode Class (modified process_fruit_detections) ---
class FruitDetectorNode(Node):
    def __init__(self, gui_message_queue):
        super().__init__('fruit_detector_node')
        self.bridge = CvBridge()
        self.gui_message_queue = gui_message_queue

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
        self.csv_file_path = os.path.join(os.path.expanduser('~'), 'fruit_detections_raw.csv') # Renamed for clarity
        self.final_output_csv_path = os.path.join(os.path.expanduser('~'), 'fruit_counting_results.csv') # Renamed for clarity
        self.csv_header = ['fruit_type', 'map_x', 'map_y', 'confidence'] # Added confidence to raw CSV
        self.initialize_csv()

        # Processing Timer
        self.processing_timer = self.create_timer(180.0, self.timer_callback) # 180 seconds = 3 minutes
        self.get_logger().info("Results processing timer set for 3 minutes.")

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
        
        # Always overwrite the final output CSV file at the start of a run
        with open(self.final_output_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['fruit_type', 'x_coordinate', 'y_coordinate']) # Match final output format
        self.get_logger().info(f"Initialized (or cleared) final results CSV file: {self.final_output_csv_path}")


    def camera_info_callback(self, msg):
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
                        
                        # Apply a confidence threshold for logging and GUI display
                        if conf >= 0.6: # Example confidence threshold. TUNE THIS!
                            self.get_logger().info(f"Detected {class_name} (conf:{conf:.2f}) at map coordinates: ({map_x:.2f}, {map_y:.2f})")
                            self.gui_message_queue.put({'fruit_type': class_name, 'map_x': map_x, 'map_y': map_y})

                            # Save to raw detections CSV
                            with open(self.csv_file_path, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([class_name, map_x, map_y, conf])
                        else:
                            self.get_logger().debug(f"Discarding detection of {class_name} (conf:{conf:.2f}) due to low confidence.")


                    except TransformException as ex:
                        self.get_logger().error(f"Could not transform point from '{msg.header.frame_id}' to 'map': {ex}")
                        
            if not results or not results[0].boxes: # Only log if no detections were found AT ALL
                self.get_logger().debug("No objects detected in current frame.") # Changed to debug for less spam

            try:
                detection_msg = self.bridge.cv2_to_imgmsg(detected_cv_image, encoding='bgr8')
                self.detection_publisher.publish(detection_msg)
            except Exception as e:
                self.get_logger().error(f"Could not convert detected image to ROS Image: {e}")

        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")

    def timer_callback(self):
        """Callback for the processing timer, triggers the final result generation."""
        self.get_logger().info("3 minutes elapsed. Initiating final fruit counting and location reporting...")
        self.process_fruit_detections()
        # self.processing_timer.cancel() # Uncomment if you only want it to run once

    def process_fruit_detections(self):
        """
        Processes raw fruit detections to generate the final CSV output.
        The goal is to provide up to 5 representative locations per fruit type
        as a safeguard for counting.
        """
        if not os.path.exists(self.csv_file_path) or os.stat(self.csv_file_path).st_size == 0:
            self.get_logger().warn("No raw fruit detection data found in CSV for processing. Skipping final output.")
            return

        try:
            df = pd.read_csv(self.csv_file_path)
            if df.empty:
                self.get_logger().warn("Raw detections CSV file is empty, no data to process. Skipping final output.")
                return

            final_output_rows = []

            # Iterate through each fruit type that was potentially detected
            for fruit_type in df['fruit_type'].unique():
                fruit_detections_for_type = df[df['fruit_type'] == fruit_type].copy()
                
                # Further optional filtering for clustering:
                # You might want to filter by a higher confidence for clustering
                # or only use detections that are stable/repeated.
                # For simplicity, we'll use all detections (above initial threshold)
                
                # Prepare points for DBSCAN
                points = fruit_detections_for_type[['map_x', 'map_y']].values

                if len(points) == 0:
                    self.get_logger().debug(f"No points for {fruit_type} after initial filtering. Skipping.")
                    continue
                
                # DBSCAN to group nearby detections of the same fruit type
                # eps: determines how close points must be to each other to be considered a part of a cluster.
                #      Adjust this based on how spatially close individual fruits on a tree might appear.
                #      E.g., 0.2 meters (20 cm) could mean fruits within 20cm of each other are in the same cluster.
                # min_samples: minimum number of points required to form a dense region (a cluster).
                #      Set to 1 or 2 if you want even small groups (or single isolated detections) to form a cluster.
                #      If a tree has many fruits, you might get several small clusters.
                db = DBSCAN(eps=0.2, min_samples=1).fit(points) # TUNE THESE!
                labels = db.labels_

                unique_clusters = set(labels)
                valid_clusters_centroids = []

                for k in unique_clusters:
                    if k == -1: # Noise points - usually discarded, but can be included if desired
                        self.get_logger().debug(f"Discarding {np.sum(labels == k)} noise points for {fruit_type}.")
                        continue
                    
                    cluster_points = points[labels == k]
                    
                    if len(cluster_points) > 0:
                        centroid_x = np.mean(cluster_points[:, 0])
                        centroid_y = np.mean(cluster_points[:, 1])
                        valid_clusters_centroids.append((centroid_x, centroid_y, len(cluster_points))) # Store centroid and size

                # Sort clusters by size (largest first) to prioritize more robust clusters
                valid_clusters_centroids.sort(key=lambda x: x[2], reverse=True)

                # Select up to 5 centroids to report for this fruit type
                # This serves as the "safeguard to ensure you actually counted the fruits"
                num_reported_points = 0
                for centroid_x, centroid_y, cluster_size in valid_clusters_centroids:
                    if num_reported_points >= 5:
                        break
                    
                    final_output_rows.append({
                        'fruit_type': fruit_type,
                        'x_coordinate': centroid_x,
                        'y_coordinate': centroid_y
                    })
                    self.get_logger().info(f"Reporting {fruit_type} cluster centroid: ({centroid_x:.2f}, {centroid_y:.2f}) (size: {cluster_size})")
                    num_reported_points += 1
                
                if num_reported_points == 0:
                    self.get_logger().warn(f"No valid clusters found to report for {fruit_type}.")


            # Save the final processed data to the CSV
            with open(self.final_output_csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['fruit_type', 'x_coordinate', 'y_coordinate'])
                for row_data in final_output_rows:
                    writer.writerow([row_data['fruit_type'], row_data['x_coordinate'], row_data['y_coordinate']])
            self.get_logger().info(f"Final fruit counting results saved to: {self.final_output_csv_path}")

        except Exception as e:
            self.get_logger().error(f"Error during final fruit counting and location reporting: {e}")

# --- Main Function (unchanged from previous response) ---
def main(args=None):
    rclpy.init(args=args)

    gui_message_queue = queue.Queue()

    node = FruitDetectorNode(gui_message_queue)

    root = tk.Tk()
    gui_app = FruitDetectionGUI(root, gui_message_queue)

    gui_thread = threading.Thread(target=root.mainloop, daemon=True)
    gui_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()