import cv2
import numpy as np
import yaml
import os
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pydobotplus import Dobot as MyDobot
from pupil_apriltags import Detector
import pyrealsense2 as rs

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
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align


def get_camera_intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy, intr.coeffs


def initialize_detector(tag_size):
    return Detector(families="tag36h11", nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0), tag_size


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
        cv2.line(color_image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)
    cv2.circle(color_image, tuple(tag.center.astype(int)), 5, (0, 0, 255), -1)
    center_x, center_y = tag.center.astype(int)
    cv2.putText(color_image, f"ID: {tag.tag_id}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    detector, tag_size = initialize_detector(0.0792)  # Set the tag size in meters 0.0792
    port = get_dobot_port()
    device = MyDobot(port=port)
    print("Homing the robotic arm for calibration...")
    print("Attach the AprilTag 15.3cm above the gripper and make the x-axes aligned before calibration.")
    device.set_home(250, 0, 50)
    device.home()

    cHt_list = []
    bHg_list = []

    print("At the camera window, press Enter to record data, press 'q' to quit and calculate the transformation.")

    while True:
        color_image, depth_image, _, _ = process_frames(pipeline, align)
        if color_image is None or depth_image is None:
            continue

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(detector, gray_image, [fx, fy, cx, cy], tag_size)

        # cHt, transforms a point in the target frame to the camera frame, which is the pose of the tag in camera frame.
        # bHg, transforms a point in the gripper frame to the base frame, which is the pose of the gripper in base frame.
        if tags:
            tag = tags[0]

            if tag.pose_t is not None:
                cHt = get_camera_to_tag_matrix(tag)
                bHg = get_robot_arm_matrix(device.get_pose())
                draw_tag_info(color_image, tag)

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("\r") or key == ord("\n"):  # Enter key
            if tags:
                # saving all the matrix
                cHt_list.append(cHt)
                bHg_list.append(bHg)
                print(f"camera to tag: \n{cHt}")
                print(f"gripper to base: \n{bHg}")
                print(f"Data recorded. Total data points: {len(cHt_list)}")

    if len(cHt_list) >= 1:
        # Inverse of cHt --> tHc
        inv_cHt_list = [np.linalg.inv(cHt) for cHt in cHt_list]

        # gripper to tag transformation (x-axes aligned)
        # [1,  0,  0,  30]
        # [0,  1,  0,   0]
        # [0,  0, -1, 153]
        # [0,  0,  0,   1]
        # tag size = 0.792

        gHt = np.array([[-1, 0, 0, 30],
                         [0, 1, 0, 0], 
                         [0, 0, -1, 153], 
                         [0, 0, 0, 1]], dtype=np.float32)
        print(gHt)
        
        bHc_list = [bHg @ gHt @ inv_cHt for bHg, inv_cHt in zip(bHg_list, inv_cHt_list)]
        avg_cHb, avg_cHb_R, avg_cHb_t, avg_rpy = get_average_transformation(bHc_list)
        print(f"Rotation Matrix:\n{avg_cHb_R}")
        print(f"Translation Vector:\n{avg_cHb_t}")
        print(avg_rpy)

        transformation_data = {"rotation": avg_cHb_R.tolist(), "translation": avg_cHb_t.tolist()}
        save_transformation(transformation_data, os.path.join(os.path.dirname(__file__), "..", "config", "camera_to_base_transformation.yaml"))
    else:
        print("Not enough data points to calculate the transformation.")

    pipeline.stop()
    device.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()