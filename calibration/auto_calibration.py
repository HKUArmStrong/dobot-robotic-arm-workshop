import cv2
import numpy as np
import yaml
import os
import sys
import time
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pydobotplus import Dobot as MyDobot
from pupil_apriltags import Detector
import pyrealsense2 as rs

# ============== CONFIGURATION CONSTANTS ==============

# Camera Configuration
CAMERA_FPS = 30

# AprilTag Configuration
TAG_SIZE_METERS = 0.0792  # 79.2mm AprilTag size in meters
TAG_FAMILY = "tag36h11"

# Gripper to Tag Transformation (mm)
# Distance from gripper to tag when x-axes are aligned
GRIPPER_TO_TAG_X = 30
GRIPPER_TO_TAG_Y = 0
GRIPPER_TO_TAG_Z = 153

# Robot Arm Home Position
HOME_X = 250
HOME_Y = 0
HOME_Z = 50

# Calibration Timing (seconds)
STABILIZATION_TIME = 1.5  # Time to wait after arm movement
DETECTION_TIMEOUT = 2.0   # Extra time to wait if tag not detected
NEXT_POSITION_DELAY = 0.5 # Delay before moving to next position

# Display Configuration
TEXT_COLOR_GREEN = (0, 255, 0)
TEXT_COLOR_YELLOW = (0, 255, 255)
TEXT_COLOR_RED = (0, 0, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE_LARGE = 0.7
TEXT_SCALE_SMALL = 0.5
TEXT_THICKNESS = 2
LINE_THICKNESS = 2
CIRCLE_RADIUS = 5

# UI Positioning
TEXT_START_Y = 30
TEXT_LINE_SPACING = 30
TEXT_START_X = 10
TAG_ID_OFFSET_Y = -10

# OpenCV Waitkey
WAITKEY_DELAY_MS = 1

# Define calibration positions (rounded from recorded poses)
calibration_positions = [
    (210, -40, 50),
    (210, -40, 80),
    (210, 60, 80),
    (210, 60, 30),
    (130, -150, 60),
    (130, -150, 0),
    (170, -25, 0),
    (250, -40, 30)
]

# ============== END CONFIGURATION ==============

def initialize_pipeline(serial=None):
    # Load camera serial from config
    config_path = os.path.join(os.path.dirname(__file__), '../config/device_port.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if serial is None:
        serial = config.get('camera_serial')
    print(f"Using camera serial: {serial}")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, CAMERA_FPS)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, CAMERA_FPS)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align


def get_camera_intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy, intr.coeffs

def initialize_detector(tag_size):
    return Detector(
        families=TAG_FAMILY, 
        nthreads=1, 
        quad_decimate=1.0, 
        quad_sigma=0.0, 
        refine_edges=1, 
        decode_sharpening=0.25, 
        debug=0
    ), tag_size


def process_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image, color_frame, depth_frame


def detect_tags(detector, gray_image, camera_params, tag_size):
    return detector.detect(gray_image, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

def draw_tag_info(color_image, tag):
    for idx in range(len(tag.corners)):
        cv2.line(
            color_image, 
            tuple(tag.corners[idx - 1, :].astype(int)), 
            tuple(tag.corners[idx, :].astype(int)), 
            TEXT_COLOR_GREEN, 
            LINE_THICKNESS
        )
    cv2.circle(color_image, tuple(tag.center.astype(int)), CIRCLE_RADIUS, TEXT_COLOR_RED, -1)
    center_x, center_y = tag.center.astype(int)
    cv2.putText(
        color_image, 
        f"ID: {tag.tag_id}", 
        (center_x, center_y + TAG_ID_OFFSET_Y), 
        TEXT_FONT, 
        TEXT_SCALE_SMALL, 
        TEXT_COLOR_GREEN, 
        TEXT_THICKNESS
    )

def get_camera_to_tag_matrix(tag):
    """
    Build a 4x4 transformation matrix (camera to tag) in mm from a detected tag object.
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = tag.pose_R
    transformation_matrix[:3, 3] = tag.pose_t.flatten() * 1000
    return transformation_matrix

def get_robot_arm_matrix(pose):
    """
    Build a 4x4 transformation matrix (base to gripper) in mm from a robot pose object.
    Assumes planar robot (rotation about Z only).
    """
    theta = np.arctan2(pose.position.y, pose.position.x)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    robot_matrix = np.eye(4)
    robot_matrix[:3, :3] = rot_z
    robot_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return robot_matrix

def get_average_transformation(mats):
    """
    Compute the average transformation (rotation and translation) from a list of 4x4 matrices.
    Returns: avg_mat, avg_R, avg_t, avg_rpy
    """
    R_mats = [mat[:3, :3] for mat in mats]
    rpys = [R.from_matrix(R_mat).as_euler("xyz", degrees=True) for R_mat in R_mats]
    avg_rpy = np.mean(rpys, axis=0)
    avg_R = R.from_euler("xyz", avg_rpy, degrees=True).as_matrix()
    t_mats = [mat[:3, 3] for mat in mats]
    avg_t = np.mean(t_mats, axis=0)
    avg_mat = np.eye(4)
    avg_mat[:3, :3] = avg_R
    avg_mat[:3, 3] = avg_t
    return avg_mat, avg_R, avg_t, avg_rpy

def save_transformation(transformation_data, filename):
    with open(filename, "w") as file:
        yaml.dump(transformation_data, file)

def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]

def main():
    pipeline, profile, align = initialize_pipeline()
    fx, fy, cx, cy, _ = get_camera_intrinsics(profile)
    detector, tag_size = initialize_detector(TAG_SIZE_METERS)
    
    print("Camera initialized. Showing camera stream...")
    print("Attach the AprilTag 15.3cm above the gripper and make the x-axes aligned before calibration.")
    print("Press SPACE to start automatic calibration, or 'q' to quit.\n")
    
    # Show camera stream before starting calibration
    start_calibration = False
    while not start_calibration:
        color_image, depth_image, _, _ = process_frames(pipeline, align)
        if color_image is None or depth_image is None:
            continue

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(detector, gray_image, [fx, fy, cx, cy], tag_size)

        if tags:
            tag = tags[0]
            if tag.pose_t is not None:
                draw_tag_info(color_image, tag)
        
        # Display instruction on the image
        cv2.putText(color_image, "Press SPACE to start calibration", (TEXT_START_X, TEXT_START_Y), 
                    TEXT_FONT, TEXT_SCALE_LARGE, TEXT_COLOR_YELLOW, TEXT_THICKNESS)
        cv2.putText(color_image, "Press 'q' to quit", (TEXT_START_X, TEXT_START_Y + TEXT_LINE_SPACING), 
                    TEXT_FONT, TEXT_SCALE_LARGE, TEXT_COLOR_YELLOW, TEXT_THICKNESS)

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(WAITKEY_DELAY_MS) & 0xFF

        if key == ord("q"):
            print("Calibration cancelled.")
            pipeline.stop()
            cv2.destroyAllWindows()
            return
        elif key == ord(" "):  # Space key
            start_calibration = True
            print("\nStarting calibration sequence...\n")
    
    port = get_dobot_port()
    device = MyDobot(port=port)
    print("Homing the robotic arm for calibration...")
    device.set_home(HOME_X, HOME_Y, HOME_Z)
    device.home()
    device.wait_for_cmd(device.move_to(HOME_X, HOME_Y, HOME_Z))

    cHt_list = []
    bHg_list = []

    print(f"\nStarting automatic calibration with {len(calibration_positions)} positions...")
    print("Press 'q' at any time to abort and calculate transformation with collected data so far.\n")

    # State variables for automatic calibration
    position_idx = 0
    move_initiated = False
    move_start_time = 0
    data_recorded_for_position = False
    
    # Continuous camera loop (like calibration.py)
    while True:
        color_image, depth_image, _, _ = process_frames(pipeline, align)
        if color_image is None or depth_image is None:
            continue

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(detector, gray_image, [fx, fy, cx, cy], tag_size)

        # Process tags and draw info
        cHt = None
        bHg = None
        if tags:
            tag = tags[0]
            if tag.pose_t is not None:
                cHt = get_camera_to_tag_matrix(tag)
                bHg = get_robot_arm_matrix(device.get_pose())
                draw_tag_info(color_image, tag)
        
        # Automatic movement logic
        if position_idx < len(calibration_positions):
            # Initiate movement to next position
            if not move_initiated:
                x, y, z = calibration_positions[position_idx]
                print(f"[{position_idx + 1}/{len(calibration_positions)}] Moving to position: ({x}, {y}, {z})...")
                device.move_to(x, y, z, wait=False)  # Non-blocking move
                move_initiated = True
                move_start_time = time.time()
                data_recorded_for_position = False
            
            # Wait for stabilization period
            elif move_initiated and (time.time() - move_start_time) > STABILIZATION_TIME:
                # Try to record data if tag is detected and not yet recorded
                if not data_recorded_for_position and tags and cHt is not None:
                    cHt_list.append(cHt)
                    bHg_list.append(bHg)
                    print(f"  ✓ Data recorded at position {position_idx + 1}. Total data points: {len(cHt_list)}")
                    data_recorded_for_position = True
                    
                    # Move to next position after a short delay
                    time.sleep(NEXT_POSITION_DELAY)
                    position_idx += 1
                    move_initiated = False
                
                # If no tag detected after waiting, move to next position anyway
                elif not data_recorded_for_position and (time.time() - move_start_time) > (STABILIZATION_TIME + DETECTION_TIMEOUT):
                    print(f"  ✗ Warning: Could not detect AprilTag at position {position_idx + 1}")
                    position_idx += 1
                    move_initiated = False
        else:
            # All positions visited, exit loop
            if position_idx >= len(calibration_positions):
                break
        
        # Display status overlay on camera feed
        status_y = TEXT_START_Y
        if position_idx < len(calibration_positions):
            status_text = f"Position {position_idx + 1}/{len(calibration_positions)}"
            if data_recorded_for_position:
                status_text += " - RECORDED"
            elif move_initiated:
                status_text += " - MOVING..."
            cv2.putText(color_image, status_text, (TEXT_START_X, status_y), 
                       TEXT_FONT, TEXT_SCALE_LARGE, TEXT_COLOR_GREEN, TEXT_THICKNESS)
            status_y += TEXT_LINE_SPACING
        
        cv2.putText(color_image, f"Total points: {len(cHt_list)}", (TEXT_START_X, status_y), 
                   TEXT_FONT, TEXT_SCALE_LARGE, TEXT_COLOR_GREEN, TEXT_THICKNESS)
        cv2.putText(color_image, "Press 'q' to finish", (TEXT_START_X, status_y + TEXT_LINE_SPACING), 
                   TEXT_FONT, TEXT_SCALE_LARGE, TEXT_COLOR_YELLOW, TEXT_THICKNESS)

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(WAITKEY_DELAY_MS) & 0xFF

        if key == ord("q"):
            print("\nUser aborted calibration.")
            break

    print(f"\nCalibration movement complete. Collected {len(cHt_list)} data points.")
    print("Calculating transformation matrix...")

    if len(cHt_list) >= 1:
        # Inverse of cHt --> tHc
        inv_cHt_list = [np.linalg.inv(cHt) for cHt in cHt_list]

        # gripper to tag transformation (x-axes aligned)
        # The transformation matrix represents the fixed relationship between
        # the gripper coordinate frame and the AprilTag coordinate frame
        gHt = np.array([
            [-1, 0, 0, GRIPPER_TO_TAG_X],
            [0, 1, 0, GRIPPER_TO_TAG_Y], 
            [0, 0, -1, GRIPPER_TO_TAG_Z], 
            [0, 0, 0, 1]
        ], dtype=np.float32)
        print(gHt)
        
        bHc_list = [bHg @ gHt @ inv_cHt for bHg, inv_cHt in zip(bHg_list, inv_cHt_list)]
        avg_cHb, avg_cHb_R, avg_cHb_t, avg_rpy = get_average_transformation(bHc_list)
        print(f"Rotation Matrix:\n{avg_cHb_R}")
        print(f"Translation Vector:\n{avg_cHb_t}")
        print(avg_rpy)

        transformation_data = {"rotation": avg_cHb_R.tolist(), "translation": avg_cHb_t.tolist()}
        save_transformation(transformation_data, os.path.join(os.path.dirname(__file__), "..", "config", "camera_to_base_transformation.yaml"))
        print("\n✓ Transformation saved to config/camera_to_base_transformation.yaml")
    else:
        print("Not enough data points to calculate the transformation.")

    pipeline.stop()
    device.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()