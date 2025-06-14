import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import cv2
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from builtin_interfaces.msg import Duration
from tf2_geometry_msgs import do_transform_point
import csv
import os

from your_detection_script import detect_glowing_mushrooms, get_contour_centroids, pixel_to_camera_coords

class MushroomLayerNode(Node):
    def __init__(self):
        super().__init__('mushroom_layer_node')

        # Parameters
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width', 200)  # 10m / 0.05 = 200 cells
        self.declare_parameter('map_height', 200)
        self.declare_parameter('map_origin_x', 0.0)
        self.declare_parameter('map_origin_y', 0.0)
        self.declare_parameter('publish_topic', '/mushroom_layer')

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.width = self.get_parameter('map_width').get_parameter_value().integer_value
        self.height = self.get_parameter('map_height').get_parameter_value().integer_value
        self.origin_x = self.get_parameter('map_origin_x').get_parameter_value().double_value
        self.origin_y = self.get_parameter('map_origin_y').get_parameter_value().double_value
        self.output_topic = self.get_parameter('publish_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        self.occupied_cells = set()
        self.csv_path = os.path.expanduser('~/mushroom_obstacles.csv')

        # ROS interfaces
        self.rgb_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.cam_info_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, self.output_topic, 10)

        self.timer = self.create_timer(1.0, self.publish_layer)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Navigation setup
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_queue = [
            (0.5, 0.5), (9.0, 9.0), (1.0, 9.0), (9.0, 1.0), (1.0, 1.0), (1.0, 5.0), (9.0, 5.0), (9.0, 1.0)
        ]
        self.current_goal_idx = 0
        self.send_next_goal()

    def cam_info_callback(self, msg):
        self.camera_info = msg

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def publish_layer(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            return

        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'map'

        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        data = [-1] * (self.width * self.height)

        mask = detect_glowing_mushrooms(self.rgb_image)
        centroids, _ = get_contour_centroids(mask, min_area=3000)

        intr = {
            'fx': self.camera_info.k[0],
            'fy': self.camera_info.k[4],
            'cx': self.camera_info.k[2],
            'cy': self.camera_info.k[5],
            'scale': 0.001
        }

        for (u, v) in centroids:
            depth = self.depth_image[v, u]
            pt = pixel_to_camera_coords(u, v, depth, intr)
            if pt is None:
                continue

            cam_point = PointStamped()
            cam_point.header.frame_id = self.camera_info.header.frame_id
            cam_point.header.stamp = self.get_clock().now().to_msg()
            cam_point.point.x = float(pt[0])
            cam_point.point.y = float(pt[1])
            cam_point.point.z = float(pt[2])

            try:
                tf_map = self.tf_buffer.lookup_transform(
                    'map',
                    cam_point.header.frame_id,
                    rclpy.time.Time()
                )
                map_point = do_transform_point(cam_point, tf_map)
                x_map = map_point.point.x
                y_map = map_point.point.y
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

            mx = int((x_map - self.origin_x) / self.resolution)
            my = int((y_map - self.origin_y) / self.resolution)

            if 0 <= mx < self.width and 0 <= my < self.height:
                idx = my * self.width + mx
                data[idx] = 100

                # Record unique cell in meters
                cell_x = self.origin_x + mx * self.resolution
                cell_y = self.origin_y + my * self.resolution
                self.occupied_cells.add((round(cell_x, 3), round(cell_y, 3)))

        grid.data = data
        self.map_pub.publish(grid)
    
        self.get_logger().info(f"Mushroom found")

        # Write unique cells to CSV
        try:
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x (m)', 'y (m)'])
                for (x, y) in sorted(self.occupied_cells):
                    writer.writerow([x, y])
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV: {e}")

    def send_next_goal(self):
        if self.current_goal_idx >= len(self.goal_queue):
            self.get_logger().info("Navigation complete.")
            return

        goal_x, goal_y = self.goal_queue[self.current_goal_idx]
        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Navigating to ({goal_x}, {goal_y})")
        self.nav_client.wait_for_server()
        self._send_goal_future = self.nav_client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Reached goal {self.current_goal_idx + 1}/{len(self.goal_queue)}")
        self.current_goal_idx += 1
        self.send_next_goal()

def main(args=None):
    rclpy.init(args=args)
    node = MushroomLayerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
 