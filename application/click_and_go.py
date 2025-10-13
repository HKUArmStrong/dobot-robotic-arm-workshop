import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import yaml
from pydobotplus import Dobot


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["clicked"] = True
        param["x"] = x
        param["y"] = y

def initialize_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align


def get_camera_intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy

def process_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image


def detect_tags(detector, gray_image, camera_params, tag_size):
    return detector.detect(gray_image, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)


def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]


def draw_tag_info(color_image, tag):
    for idx in range(len(tag.corners)):
        cv2.line(color_image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)
    cv2.circle(color_image, tuple(tag.center.astype(int)), 5, (0, 0, 255), -1)
    center_x, center_y = tag.center.astype(int)
    cv2.putText(color_image, f"ID: {tag.tag_id}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def load_transformation(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    R = np.array(data["rotation"], dtype=np.float32)
    t = np.array(data["translation"], dtype=np.float32).reshape(3, 1)
    return R, t

def main():
    pipeline, profile, align = initialize_pipeline()
    fx, fy, cx, cy = get_camera_intrinsics(profile)
    R_camera2base, t_camera2base = load_transformation(os.path.join(os.path.dirname(__file__), "..", "config", "camera_to_base_transformation.yaml"))
    
    print(R_camera2base)
    print(t_camera2base)

    port = get_dobot_port()
    device = Dobot(port=port)

    print("Homing the robotic arm for calibration...")
    device.set_home(250, 0, 50)
    # Raise up the arm to avoid collisions
    device.move_to(250, 0, 50, 0)  # Move to a safe position
    device.home()

    param = {"clicked": False, "x": 0, "y": 0}
    cv2.namedWindow("Click and Go")
    cv2.setMouseCallback("Click and Go", click_event, param)

    print("Click on a point in the window to move the robot to that point. Press 'q' to quit.")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        cv2.imshow("Click and Go", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if param["clicked"]:
            x, y = param["x"], param["y"]
            print(x,y)

            pc = rs.pointcloud()
            pc.map_to(color_frame)
            obj_coordinate = depth_frame.get_distance(x, y)
            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], obj_coordinate)
            point_coordinate = np.array(point).reshape(3,1) # Reshape to (1, 3)
            point_coordinate *= 1000
            print(t_camera2base.shape)
            target_coordinate = R_camera2base @ point_coordinate + t_camera2base.reshape(3,1)
            print(target_coordinate.shape)
            print(f"Calculated camera coordinate (mm): {point_coordinate.flatten()}")
            print(f"Calculated base coordinate (mm): {target_coordinate.flatten()}")
            device.move_to(target_coordinate[0], target_coordinate[1], target_coordinate[2]+10)
            device.clear_alarms()

            param["clicked"] = False

    pipeline.stop()
    device.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()