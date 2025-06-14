import rclpy
from rclpy.node import Node
from math import sin, cos, pi
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Joy, Image, CameraInfo
from tf_transformations import quaternion_from_euler
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.action.client import ClientGoalHandle
from cv_bridge import CvBridge
import cv2
import torch
import os
from ultralytics import YOLO
import numpy as np
import csv
import time

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point
import geometry_msgs.msg
import rclpy.executors

# --- FruitDetectorNode Class (Without Clustering) ---
# This class handles image processing, fruit detection, coordinate transformation,
# and saving individual detection data. It is designed to be activated and deactivated
# by a parent node (like TreeNavigator).
class FruitDetectorNode(Node):
    def __init__(self, parent_node_name=None, detection_complete_callback=None):
        # Create a unique node name, possibly based on the parent's name
        if parent_node_name:
            super().__init__(f'{parent_node_name}_fruit_detector')
        else:
            super().__init__('fruit_detector_node')

        self.bridge = CvBridge()
        self._is_active = False # Flag to control if the detector is actively processing images
        self.detection_complete_callback = detection_complete_callback # Callback for parent node on completion

        # Subscriptions are initially None; they will be created/destroyed dynamically
        self.image_subscription = None
        self.camera_info_subscription = None

        # Publisher for detected images (with bounding boxes)
        self.detection_publisher = self.create_publisher(
            Image,
            'detected_image',
            10
        )

        # YOLO Model Loading
        try:
            # Construct path to the YOLO model weights relative to the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(script_dir, 'weights', 'fruit_detector_v1.pt')
            self.model = YOLO(weights_path)
            self.model.eval() # Set the model to evaluation mode
            self.get_logger().info("YOLOv11 (Fast) model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Error loading YOLOv11 (Fast) model: {e}")

        # TF2 Buffer and Listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera Intrinsics - will be populated by camera_info_callback
        self.camera_intrinsics = None

        # --- Image Saving Setup ---
        # Directory to save cropped images of detected fruits
        self.image_save_directory = os.path.join(os.path.expanduser('~'), 'fruit_detection_images')
        os.makedirs(self.image_save_directory, exist_ok=True) # Create directory if it doesn't exist
        self.get_logger().info(f"Saving detected images to: {self.image_save_directory}")

        # CSV File Setup for raw detections
        self.csv_file_path = os.path.join(os.path.expanduser('~'), 'fruit_detections.csv')
        # Header for the raw detections CSV, including image filename
        self.csv_header = ['timestamp', 'fruit_type', 'map_x', 'map_y', 'image_filename']
        self.initialize_csv_file() # Initialize CSV file with header

        # Detection Timer - initially None, will be created/destroyed dynamically
        self.detection_timer = None

        self.get_logger().info("Fruit Detector Node Initialized (inactive).")

    def initialize_csv_file(self):
        """Initializes the raw detections CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.csv_header)
            self.get_logger().info(f"Created new raw detections CSV file: {self.csv_file_path}")
        else:
            self.get_logger().info(f"Appending to existing raw detections CSV file: {self.csv_file_path}")

    def _create_subscriptions(self):
        """Creates the image and camera info subscriptions only when detection is active."""
        if self.image_subscription is None:
            self.image_subscription = self.create_subscription(
                Image,
                '/camera/camera/color/image_raw',
                self.image_callback,
                10
            )
            self.get_logger().info("Image subscription created.")
        
        if self.camera_info_subscription is None:
            # Use a special one-time callback for camera info to get intrinsics
            self.camera_info_subscription = self.create_subscription(
                CameraInfo,
                '/camera/camera/color/camera_info',
                self._camera_info_callback_once,
                10
            )
            self.get_logger().info("Camera info subscription created.")

    def _destroy_subscriptions(self):
        """Destroys the image and camera info subscriptions when detection stops."""
        if self.image_subscription is not None:
            self.destroy_subscription(self.image_subscription)
            self.image_subscription = None
            self.get_logger().info("Image subscription destroyed.")
        
        # The camera_info_subscription might have already been destroyed by _camera_info_callback_once
        if self.camera_info_subscription is not None:
            self.destroy_subscription(self.camera_info_subscription)
            self.camera_info_subscription = None
            self.get_logger().info("Camera info subscription destroyed.")

    def _camera_info_callback_once(self, msg):
        """
        Stores camera intrinsic parameters and then destroys its own subscription.
        This ensures camera info is only read once.
        """
        self.camera_intrinsics = msg.k
        self.get_logger().info("Received camera info. Destroying camera info subscription.")
        # Destroy the subscription immediately after receiving the info
        self.destroy_subscription(self.camera_info_subscription)
        self.camera_info_subscription = None # Clear the reference

    def start_detection(self):
        """Activates the fruit detection process, clearing old data and starting subscriptions/timer."""
        if not self._is_active:
            self.get_logger().info("Starting fruit detection...")
            self._is_active = True
            self._create_subscriptions() # Create (and activate) image and camera info subscriptions

            # Start or reset the detection timer.
            # Using a shorter timer (10 seconds) for faster testing cycles.
            # For real deployment, revert to 180.0 seconds (3 minutes).
            if self.detection_timer is not None:
                self.detection_timer.cancel() # Cancel existing timer if any
            self.detection_timer = self.create_timer(5.0, self.timer_callback)
            self.get_logger().info("Detection timer started for 10 seconds (for testing).")
            # For production: self.detection_timer = self.create_timer(180.0, self.timer_callback)

    def stop_detection(self):
        """Deactivates the fruit detection process, stopping subscriptions and timers."""
        if self._is_active:
            self.get_logger().info("Stopping fruit detection...")
            self._is_active = False
            self._destroy_subscriptions() # Destroy (and deactivate) subscriptions
            if self.detection_timer is not None:
                self.detection_timer.cancel() # Cancel the detection timer
                self.detection_timer = None # Clear the timer reference
            self.get_logger().info("Fruit detection stopped and timer cancelled.")

    def image_callback(self, msg):
        """
        Callback for incoming image messages. Performs fruit detection,
        transforms coordinates, saves images, and stores data for the CSV.
        """
        if not self._is_active:
            # Only process images if the detector is actively enabled
            return

        if self.camera_intrinsics is None:
            # If camera info hasn't been received yet, warn and wait.
            self.get_logger().warn("Skipping image callback: Camera intrinsics not received yet. Waiting for info.")
            return

        CLASS_NAMES = ["Apple", "Banana", "Grapes", "Lemon", "Orange"]

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detected_cv_image = cv_image.copy() # Make a copy to draw bounding boxes on

        current_timestamp = time.time() # Get current timestamp for unique filenames
        
        try:
            # Lookup transform from camera frame to map frame
            transform = self.tf_buffer.lookup_transform(
                'map',                  # Target frame
                msg.header.frame_id,    # Source frame (camera)
                rclpy.time.Time(),      # Lookup at the latest available transform
                timeout=rclpy.duration.Duration(seconds=0.1) # Timeout for the transform lookup
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not transform '{msg.header.frame_id}' to 'map': {ex}")
            return

        try:
            # Perform inference with the YOLO model
            results = self.model(rgb_image)

            if results and results[0].boxes: # Check if any detections were made
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) # Bounding box coordinates
                confidences = results[0].boxes.conf.cpu().numpy() # Confidence scores
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int) # Class IDs

                self.get_logger().info(f"Detected {len(boxes)} objects")

                # Process each detected object
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    conf = confidences[i]
                    cls_id = class_ids[i]

                    # Get class name from CLASS_NAMES list
                    class_name = f"Class {cls_id}"
                    if 0 <= cls_id < len(CLASS_NAMES):
                        class_name = CLASS_NAMES[cls_id]

                    # Draw bounding box and label on the image
                    label = f"{class_name}: {conf:.2f}"
                    color = (0, 255, 0) # Green color for bounding box
                    cv2.rectangle(detected_cv_image, (x1, y1), (x2, y2), color, 2)
                    # Adjust text position to be above the box
                    cv2.putText(detected_cv_image, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Calculate the center of the bounding box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Assume a depth for 3D point estimation (adjust if a depth sensor is available)
                    assumed_depth = 1.0 # meters - ADJUST THIS IF YOU HAVE A BETTER ESTIMATE OR DEPTH SENSOR
                    
                    # Get camera intrinsics
                    fx = self.camera_intrinsics[0]
                    fy = self.camera_intrinsics[4]
                    cx = self.camera_intrinsics[2]
                    cy = self.camera_intrinsics[5]

                    # Convert 2D pixel coordinates to 3D point in camera frame
                    point_camera_x = (center_x - cx) * assumed_depth / fx
                    point_camera_y = (center_y - cy) * assumed_depth / fy
                    point_camera_z = assumed_depth # Assumed depth is the Z-coordinate

                    # Create a PointStamped message for transformation
                    point_in_camera_frame = geometry_msgs.msg.PointStamped()
                    point_in_camera_frame.header.frame_id = msg.header.frame_id
                    point_in_camera_frame.header.stamp = msg.header.stamp
                    point_in_camera_frame.point.x = point_camera_x
                    point_in_camera_frame.point.y = point_camera_y
                    point_in_camera_frame.point.z = point_camera_z

                    try:
                        # Transform the 3D point from camera frame to map frame
                        point_in_map_frame = do_transform_point(point_in_camera_frame, transform)
                        map_x = point_in_map_frame.point.x
                        map_y = point_in_map_frame.point.y
                        
                        self.get_logger().info(f"Detected {class_name} at map coordinates: ({map_x:.2f}, {map_y:.2f})")
                        
                        # --- Save cropped image for this specific detection ---
                        # Create a unique filename for the cropped detected image
                        image_filename = f"{map_x:.2f}_{map_y:.2f}_{class_name}_{i}.png"
                        image_filepath = os.path.join(self.image_save_directory, image_filename)
                        
                        # Clip bounding box coordinates to ensure they are within image bounds
                        x1_clip = max(0, x1)
                        y1_clip = max(0, y1)
                        x2_clip = min(cv_image.shape[1], x2)
                        y2_clip = min(cv_image.shape[0], y2)
                        
                        # Crop the image to the bounding box region
                        cropped_image = cv_image[y1_clip:y2_clip, x1_clip:x2_clip]
                        
                        # Save the cropped image if it's not empty
                        if cropped_image.size > 0:
                            cv2.imwrite(image_filepath, detected_cv_image)
                            self.get_logger().info(f"Saved detection image: {image_filepath}")
                        else:
                            self.get_logger().warn(f"Image for {class_name} at ({x1},{y1})-({x2},{y2}) was empty.")
                            image_filename = "" # Don't record a filename if image wasn't saved

                        # Save to raw detections CSV file immediately
                        with open(self.csv_file_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([class_name, map_x, map_y, image_filename])

                    except TransformException as ex:
                        self.get_logger().error(f"Could not transform point from '{msg.header.frame_id}' to 'map': {ex}")
                        
            else:
                self.get_logger().info("No objects detected.")

            # Publish the image with bounding boxes drawn on it for visualization
            try:
                detection_msg = self.bridge.cv2_to_imgmsg(detected_cv_image, encoding='bgr8')
                self.detection_publisher.publish(detection_msg)
            except Exception as e:
                self.get_logger().error(f"Could not convert detected image to ROS Image: {e}")

        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")

    def timer_callback(self):
        """
        Callback for the detection timer. This method signals the completion
        of the detection period and then calls the parent node's callback.
        """
        self.get_logger().info("Detection timer elapsed. Stopping detection.")
        self.stop_detection() # Stop the fruit detector (subscriptions, timer)

        # Call the callback function provided by the parent node (TreeNavigator)
        # This allows the TreeNavigator to resume its navigation logic.
        if self.detection_complete_callback:
            self.get_logger().info("Calling detection complete callback for parent node.")
            self.detection_complete_callback()

# --- TreeNavigator Class ---
# This class manages the robot's navigation mission, moving between predefined
# tree positions and triggering fruit detection at each goal point.
class TreeNavigator(Node):
    def __init__(self):
        super().__init__('tree_navigator')

        # Publisher for initial pose (important for Nav2 initialization)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        )
        # self.publish_initial_pose(1.0, 1.0, 0.0) # Publish an initial pose for the robot

        # Subscribe to joystick topic for pause/resume functionality
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        # Create action client for Nav2's NavigateToPose action
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_to_pose_client.wait_for_server() # Wait until Nav2 action server is available
        self.get_logger().info('Nav2 Action server available.')

        # Define pre-programmed tree positions the robot will visit
        self.tree_positions = [
            {'fruit': 'tree_1', 'x': 3.0, 'y': 3.2},
            {'fruit': 'tree_2', 'x': 2.0, 'y': 6.85},
            {'fruit': 'tree_3', 'x': 4.75, 'y': 6.2},
            {'fruit': 'tree_4', 'x': 7.6, 'y': 2.15},
            {'fruit': 'tree_5', 'x': 8.0, 'y': 7.15}
        ]

        self.circle_radius = 0.5 # Radius of the circle around each tree for waypoints
        self.num_points = 3 # Number of waypoints to generate around each tree

        # State management flags for the mission
        self.paused = True # Mission starts in a paused state
        self.mission_started = False # Indicates if the mission has begun
        self.current_tree_idx = 0 # Index of the current tree being visited
        self.current_point_idx = 0 # Index of the current waypoint around the tree
        self.current_goal_handle: ClientGoalHandle = None # Handle for the active navigation goal
        self.goal_active = False # True if a navigation goal is actively being pursued
        self.is_detecting_fruits = False # New flag: True if fruit detection is currently active

        # Instantiate the FruitDetectorNode, passing a callback method to be called
        # when the fruit detection cycle is complete.
        self.fruit_detector = FruitDetectorNode(
            parent_node_name=self.get_name(), # Pass current node's name for unique detector node naming
            detection_complete_callback=self.on_fruit_detection_complete # Register callback
        )

        # Timer to periodically execute mission steps (e.g., send next goal)
        self.timer = self.create_timer(0.5, self.execute_mission_step)
        self.get_logger().info('Mission is paused. Press resume button (button 0) to start.')

    def publish_initial_pose(self, x, y, yaw):
        """Publishes the initial pose of the robot on the map."""
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        initial_pose.pose.pose.position.z = 0.0

        # Convert yaw (Euler angle) to a quaternion for orientation
        quat = quaternion_from_euler(0, 0, yaw)
        initial_pose.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        
        # Standard covariance matrix (example values, tune as needed)
        initial_pose.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0685, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0685, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0685, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685
        ]
        self.initial_pose_pub.publish(initial_pose)
        self.get_logger().info(f"Published initial pose at x:{x}, y:{y}, yaw:{yaw}")

    def joy_callback(self, msg: Joy):
        """
        Callback for joystick messages, used to pause or resume the mission.
        Button 0 (A) for resume, Button 1 (B) for pause (common Xbox/Logitech mapping).
        """
        if len(msg.buttons) > 1:
            if msg.buttons[1] == 1: # Button 1 (B) usually maps to pause
                if not self.paused:
                    self.get_logger().info("Pause button pressed. Pausing mission.")
                    self.pause_mission()
            elif msg.buttons[0] == 1: # Button 0 (A) usually maps to resume
                if self.paused:
                    self.get_logger().info("Resume button pressed. Resuming mission.")
                    self.resume_mission()

    def pause_mission(self):
        """Pauses the navigation mission and cancels any active goals."""
        self.paused = True
        # Cancel current navigation goal if one is active
        if self.current_goal_handle is not None and self.goal_active:
            cancel_future = self.current_goal_handle.cancel_goal_async()
            self.get_logger().info("Navigation goal cancellation requested due to pause.")
            self.goal_active = False # Assume goal is no longer active after cancellation request
        
        # Stop fruit detection if it's currently running
        if self.is_detecting_fruits:
            self.fruit_detector.stop_detection()
            self.is_detecting_fruits = False # Reset flag

    def resume_mission(self):
        """Resumes the navigation mission from its current state."""
        self.paused = False
        if not self.mission_started:
            self.get_logger().info('Mission starting from the beginning.')
            self.mission_started = True
            self.current_tree_idx = 0
            self.current_point_idx = 0
            # Generate waypoints for the first tree immediately upon mission start
            self.poses = self.generate_poses_around_tree(self.tree_positions[self.current_tree_idx])

        self.get_logger().info("Mission resumed.")


    def execute_mission_step(self):
        """
        Periodically checks mission status and sends new navigation goals.
        This is the main loop for the navigation mission.
        """
        if self.paused or not self.mission_started:
            # Don't send new goals if mission is paused or hasn't explicitly started
            return

        # Do not send new goals or advance if a navigation goal is active OR if fruit detection is active
        if self.goal_active or self.is_detecting_fruits:
            return

        # Check if the entire mission is complete
        if self.current_tree_idx >= len(self.tree_positions):
            self.get_logger().info('Mission complete. All trees visited!')
            self.timer.cancel() # Stop the mission timer
            return

        # If all points around the current tree have been visited, move to the next tree
        if self.current_point_idx >= len(self.poses):
            self.get_logger().info(f'Finished navigating around {self.tree_positions[self.current_tree_idx]["fruit"]} tree.')
            self.current_tree_idx += 1 # Advance to the next tree
            self.current_point_idx = 0 # Reset point index for the new tree
            
            # Check if there are more trees to visit
            if self.current_tree_idx < len(self.tree_positions):
                self.get_logger().info(f'Navigating around {self.tree_positions[self.current_tree_idx]["fruit"]} tree.')
                # Generate new waypoints for the next tree
                self.poses = self.generate_poses_around_tree(self.tree_positions[self.current_tree_idx])
            else:
                # No more trees, the mission will complete on the next loop iteration
                return

        # Get the current waypoint to navigate to
        current_pose = self.poses[self.current_point_idx]
        self.get_logger().info(f'Sending goal {self.current_point_idx + 1}/{len(self.poses)} around {self.tree_positions[self.current_tree_idx]["fruit"]} tree at ({current_pose.pose.position.x:.2f}, {current_pose.pose.position.y:.2f})')

        # Send the navigation goal
        self.send_nav_goal_async(current_pose)
        self.goal_active = True # Set flag to indicate a navigation goal is in progress

    def generate_poses_around_tree(self, tree):
        """
        Generates a set of circular waypoints around a given tree position.
        The robot will visit these points to get different perspectives of the tree.
        """
        poses = []
        for i in range(self.num_points):
            angle = 2 * pi * i / self.num_points
            # Calculate (x,y) coordinates on a circle around the tree
            x = tree['x'] + self.circle_radius * cos(angle)
            y = tree['y'] + self.circle_radius * sin(angle)
            # Calculate yaw to face the tree
            yaw = (angle + pi) % (2 * pi) # Robot faces towards the center of the circle

            # Create a PoseStamped message for the waypoint
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            # Set orientation using the calculated yaw
            quat = quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            poses.append(pose)
        return poses

    def send_nav_goal_async(self, pose):
        """Sends an asynchronous navigation goal to the Nav2 action server."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        # Send the goal and attach a callback for when the goal is accepted/rejected
        self.nav_to_pose_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Callback executed when a navigation goal is accepted or rejected by Nav2."""
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected by action server.')
            self.goal_active = False # Clear goal active flag
            self.current_point_idx += 1 # Move to the next point if goal was rejected
            return

        self.get_logger().info('Navigation goal accepted.')
        self.current_goal_handle = goal_handle
        # Attach a callback for when the goal result is received (success/failure/cancel)
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Callback executed when the result of a navigation goal is received."""
        result = future.result().result
        status = future.result().status

        self.current_goal_handle = None # Clear the goal handle
        self.goal_active = False # Clear the goal active flag

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation goal succeeded! Robot has reached the target waypoint.')
            # Trigger fruit detection **only when navigation goal is successfully reached**
            self.trigger_fruit_detection(self.tree_positions[self.current_tree_idx], self.poses[self.current_point_idx])
            # Do NOT increment current_point_idx here. It will be incremented AFTER detection completes.
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Navigation goal was cancelled.')
            # If cancelled (e.g., by pause), we don't advance the waypoint, allowing retry on resume
        else:
            self.get_logger().warn(f'Navigation goal failed with status code: {status}')
            self.current_point_idx += 1 # Move to next point if navigation failed

    def trigger_fruit_detection(self, tree, pose):
        """
        Initiates the fruit detection process by activating the FruitDetectorNode.
        Sets a flag to prevent navigation from proceeding until detection is done.
        """
        self.get_logger().info(f"Triggering fruit detection at x: {pose.pose.position.x:.2f}, y: {pose.pose.position.y:.2f} for {tree['fruit']} tree.")
        self.is_detecting_fruits = True # Set flag: detection is now active
        self.fruit_detector.start_detection() # Tell the detector node to start
        self.get_logger().info("Fruit detector is now active. Mission execution paused until detection completes.")

    def on_fruit_detection_complete(self):
        """
        Callback method provided to FruitDetectorNode.
        This method is called by FruitDetectorNode when its detection period finishes.
        """
        self.get_logger().info("Fruit detection cycle completed. Resuming mission navigation.")
        self.is_detecting_fruits = False # Clear flag: detection is no longer active
        self.current_point_idx += 1 # Now that detection is done, move to the next navigation waypoint
        # The next call to execute_mission_step will send the next goal


def main(args=None):
    rclpy.init(args=args)
    
    # Use a MultiThreadedExecutor to allow both the TreeNavigator and
    # the FruitDetectorNode's callbacks to run concurrently.
    executor = rclpy.executors.MultiThreadedExecutor()
    
    navigator = TreeNavigator()
    # Add both the navigator node and its internal fruit_detector node to the executor
    executor.add_node(navigator)
    executor.add_node(navigator.fruit_detector) # It's crucial to add the detector node too

    try:
        executor.spin() # Spin the executor to run all added nodes' callbacks
    except KeyboardInterrupt:
        pass # Handle Ctrl+C gracefully
    finally:
        # Ensure all nodes are properly destroyed on shutdown
        navigator.destroy_node()
        navigator.fruit_detector.destroy_node()
        rclpy.shutdown() # Shut down ROS 2

if __name__ == '__main__':
    main()