import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm # Added for progress bar

# --- Camera Intrinsics (Constant) ---
# These values are specific to the camera used to capture the images.
# fx, fy: focal lengths in pixels
# cx, cy: principal point (camera center) in pixels
# scale: converts depth map values (e.g., from millimeters) to meters.
CAMERA_INTRINSICS = {
    "fx": 616.36529541,
    "fy": 616.20294189,
    "cx": 319.5,
    "cy": 239.5,
    "scale": 0.001
}

LIGHTS_ON_LOWER = np.array([0, 0, 40])
LIGHTS_ON_UPPER = np.array([180, 140, 255])
LIGHTS_OFF_LOWER = np.array([0, 0, 60])
LIGHTS_OFF_UPPER = np.array([30, 175, 255])

def detect_glowing_mushrooms(rgb_img,lower=LIGHTS_ON_LOWER, upper=LIGHTS_ON_UPPER):
    """
    Detects glowing mushrooms in an RGB image using color thresholding in HSV space.
    
    Args:
        rgb_img: The input BGR image from OpenCV.
    
    Returns:
        A binary mask where white pixels represent detected mushroom areas.
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # Mask HSV range for the glowing yellow/orange color of the mushrooms
    mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_contour_centroids(mask, min_area=100):
    """
    Finds contours in a mask and calculates their centroids.
    
    Args:
        mask: The binary input mask.
        min_area: The minimum area for a contour to be considered valid.
    
    Returns:
        A tuple containing:
        - centroids: A list of (x, y) pixel coordinates for each valid contour.
        - valid_contours: A list of the contour objects themselves.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            valid_contours.append(cnt)
    return centroids, valid_contours

def pixel_to_camera_coords(u, v, depth, intr):
    """
    Converts a 2D pixel coordinate with depth into a 3D point in the camera's coordinate system.
    
    Args:
        u: The horizontal pixel coordinate (x).
        v: The vertical pixel coordinate (y).
        depth: The depth value at (u, v) from the depth map.
        intr: The camera intrinsics dictionary.
    
    Returns:
        A numpy array [x, y, z] representing the 3D point, or None if depth is invalid.
    """
    if depth <= 0: return None
    z = depth * intr["scale"]
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.array([x, y, z])

def resize_with_aspect_ratio(image, width=None, height=None):
    """
    Resizes an image while maintaining its aspect ratio.
    The output image will fit within the specified width and height.
    """
    if width is None and height is None:
        return image
    h, w = image.shape[:2]
    if width is None:
        scale = height / h
        width = int(w * scale)
    elif height is None:
        scale = width / w
        height = int(h * scale)
    else:
        # If both are provided, scale to fit within the smaller dimension
        scale = min(width / w, height / h)
        width = int(w * scale)
        height = int(h * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def process_frame(rgb_img, depth_img, intrinsics, min_area=100):
    """
    Processes a single RGB frame (and optional depth frame) to find mushrooms.
    
    Args:
        rgb_img: The input BGR image.
        depth_img: The corresponding depth image (can be None).
        intrinsics: The camera intrinsics dictionary.
        min_area: Minimum contour area to be considered a detection.
        
    Returns:
        A tuple containing:
        - out: The visualized output image with overlays.
        - poses: A list of dictionaries, each with pixel and 3D pose information.
    """
    mask = detect_glowing_mushrooms(rgb_img)
    centroids, valid_contours = get_contour_centroids(mask, min_area)
    poses = []
    for (u, v) in centroids:
        pose = {"pixel": (u, v), "pose_m": None}
        if depth_img is not None:
            depth = depth_img[v, u]
            pt = pixel_to_camera_coords(u, v, depth, intrinsics)
            if pt is not None:
                pose["pose_m"] = pt
        poses.append(pose)

    # Create visualization overlay on a copy of the RGB image
    out = rgb_img.copy()
    cv2.drawContours(out, valid_contours, -1, (0,255,0), 2)
    for pose in poses:
        u, v = pose["pixel"]
        cv2.circle(out, (u, v), 8, (0, 255, 255), 2)
        if pose["pose_m"] is not None:
            label = f"({pose['pose_m'][0]:.2f}, {pose['pose_m'][1]:.2f}, {pose['pose_m'][2]:.2f})m"
        else:
            label = "(no depth)"
        cv2.putText(out, label, (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    # Resize for consistent display and video saving
    out = resize_with_aspect_ratio(out, width=1280, height=720)
    return out, poses

def process_folder(img_folder, min_area=3000):
    """
    Processes all images in a folder.
    """
    rgb_paths = []
    for ext in ["png", "jpg", "jpeg"]:
        rgb_paths += glob.glob(os.path.join(img_folder, f"*.{ext}"))
    rgb_paths = [p for p in rgb_paths if "_depth" not in os.path.basename(p).lower()]

    for rgb_path in rgb_paths:
        depth_img = None
        base_noext = os.path.splitext(os.path.basename(rgb_path))[0]
        # Look for a corresponding depth image
        for ext in ["png", "jpg", "jpeg"]:
            testname = os.path.join(img_folder, f"{base_noext}_depth.{ext}")
            if os.path.exists(testname):
                depth_img = cv2.imread(testname, -1) # Read with full depth
                break

        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print("Failed to load:", rgb_path)
            continue

        vis, poses = process_frame(rgb_img, depth_img, CAMERA_INTRINSICS, min_area)

        print(f"\nFile: {os.path.basename(rgb_path)}")
        for pose in poses:
            print(" - pixel:", pose["pixel"],
                  "pose_m:" if pose["pose_m"] is not None else "(no depth)",
                  pose["pose_m"] if pose["pose_m"] is not None else "")

        cv2.imshow("Detections Overlay", vis)
        k = cv2.waitKey(0)
        if k == 27: # ESC key to break
            break
    cv2.destroyAllWindows()

def process_video(video_path, min_area=3000):
    """
    Processes a video file frame by frame, shows a progress bar, and saves the output.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    # Get video properties for progress bar and output writer
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the output video path and writer
    # The output resolution is based on the resize in `process_frame`
    output_resolution = (1280, 720)
    output_filename = os.path.splitext(video_path)[0] + "_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, output_resolution)
    
    print(f"Processing video. Press ESC in the display window to quit early.")
    
    # Use tqdm for a progress bar
    for _ in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            print("\nReached end of video or failed to read frame.")
            break

        # No depth available from video; pass None for depth_img
        vis, _ = process_frame(frame, None, CAMERA_INTRINSICS, min_area)

        # Write the processed frame to the output video file
        out_writer.write(vis)

        cv2.imshow("Detections Overlay", vis)
        # Use waitKey(1) for video processing to allow display refresh and key checks
        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            print("\nProcessing stopped by user.")
            break
            
    print(f"\nFinished processing. Video saved as '{output_filename}'")
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

def main():
    # You can modify these default values
    img_folder = "lights/off"
    video_path = None  # Or provide video file path here, e.g. "videos/mushrooms.mp4"
    min_area = 3000

    # Check for a command line argument to override the defaults
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg) and arg.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = arg
            img_folder = None # Prioritize video if a valid file is given
        elif os.path.isdir(arg):
            img_folder = arg
            video_path = None # Prioritize folder if a valid one is given
        else:
            print(f"Argument '{arg}' not recognized as a video or folder. Using defaults.")

    if video_path:
        process_video(video_path, min_area)
    elif img_folder:
        if not os.path.isdir(img_folder):
             print(f"Error: Folder '{img_folder}' not found.")
             return
        process_folder(img_folder, min_area)
    else:
        print("No video or image folder specified. Please edit the 'main' function.")


if __name__ == "__main__":
    # If you don't have tqdm, you can install it via pip:
    # pip install opencv-python numpy tqdm
    main()

