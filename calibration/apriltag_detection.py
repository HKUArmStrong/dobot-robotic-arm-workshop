# Need to continuously output the apriltag detection results (pose)

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import os
import sys
from pupil_apriltags import Detector

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
    return color_image, depth_image

def detect_tags(detector, gray_image, camera_params, tag_size):
    return detector.detect(gray_image, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

def draw_green_box(color_image, tag):
    """Draw a green box around the AprilTag without displaying the ID."""
    for idx in range(len(tag.corners)):
        cv2.line(color_image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)

def main():
    pipeline, profile, align = initialize_pipeline()
    fx, fy, cx, cy, _ = get_camera_intrinsics(profile)
    detector, tag_size = initialize_detector(0.0792)  # Set the tag size in meters

    print("Displaying camera stream with AprilTag detection. Press 'q' to quit.")

    while True:
        color_image, depth_image = process_frames(pipeline, align)
        if color_image is None or depth_image is None:
            continue

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(detector, gray_image, [fx, fy, cx, cy], tag_size)

        if tags:
            for tag in tags:
                draw_green_box(color_image, tag)
                print(f"Detected tag ID: {tag.tag_id}, Center: {tag.center}, Pose (rvec, tvec): ({tag.pose_R}, {tag.pose_t})")

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
