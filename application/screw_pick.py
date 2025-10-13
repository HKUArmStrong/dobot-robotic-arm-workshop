import os
import time
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from ultralytics import YOLO
from pydobotplus import Dobot


BASE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "config"))
MODEL_PATH = os.path.join(BASE_DIR, "../demo.pt")
LABEL_MAP = {0: "m8 nut", 1: "m8 screw", 2: "m4 screw"}
COLOR_MAP = {0: (195, 0, 0), 1: (0, 84, 160), 2: (0, 146, 36)}


class CameraThread(threading.Thread):
    def __init__(self, model_path, label_map, color_map, detection_area=None):
        super().__init__()
        print("Loading YOLO model...")
        self.model = YOLO(model_path, verbose=False)
        print("Model loaded.")
        self.label_map = label_map
        self.color_map = color_map
        self.detection_area = detection_area
        self.pipeline = None
        self.align = None
        self.running = True
        self.detecting = True
        self.frame_queue = []
        self.detection_results = []

    def initialize_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def run(self):
        print("Camera thread starting...")
        self.initialize_camera()
        print("Camera initialized.")
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            display_image = color_image.copy()

            # Draw detection area if configured
            if self.detection_area:
                cv2.polylines(display_image, [np.array(self.detection_area, np.int32)], True, (0, 255, 255), 2)

            if self.detecting:
                results = self.model.predict(color_image, task='obb', verbose=False)
                detections = []
                for result in results:
                    if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                        for obb in result.obb:
                            points = obb.xyxyxyxy[0].cpu().numpy()
                            conf = obb.conf[0].cpu().numpy()
                            cls = int(obb.cls[0].cpu().numpy())
                            label = self.label_map.get(cls, "unknown")
                            color = self.color_map.get(cls, (255, 255, 255))
                            center = np.mean(points.reshape(-1, 2), axis=0)
                            center_x, center_y = center

                            # Get 3D point
                            depth_value = depth_frame.get_distance(int(center_x), int(center_y))
                            point_3d = None
                            if depth_value > 0:
                                point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [center_x, center_y], depth_value)
                                point_3d = np.array(point) * 1000  # to mm

                            detections.append({
                                'points': points,
                                'conf': conf,
                                'label': label,
                                'color': color,
                                'center': center,
                                'point_3d': point_3d
                            })

                            # Draw the oriented bounding box
                            pts = points.reshape((-1, 1, 2)).astype(np.int32)
                            cv2.polylines(display_image, [pts], True, color, 2)
                            cv2.putText(display_image, f"{label} {conf:.2f}", (int(center[0]), int(center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        pass
                # Filter detections to only those in detection area
                if self.detection_area:
                    contour = np.array(self.detection_area, np.int32)
                    detections = [d for d in detections if cv2.pointPolygonTest(contour, tuple(d['center']), False) >= 0]
                # Store latest detections
                self.detection_results = detections
            else:
                # Keep last detections for potential use
                pass

            # Display the image
            cv2.imshow("Screw Detection", display_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.pipeline.stop()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False


def get_dobot_port():
    config_file = os.path.join(CONFIG_DIR, "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]


def load_transformation(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    R = np.array(data["rotation"], dtype=np.float32)
    t = np.array(data["translation"], dtype=np.float32).reshape(3, 1)
    return R, t


def load_detection_area(filename):
    try:
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        return data.get("detection_area")
    except FileNotFoundError:
        return None


def main():
    # Initialize Dobot
    port = get_dobot_port()
    device = Dobot(port=port)
    device.home()

    # Load transformation
    R_camera2base, t_camera2base = load_transformation(os.path.join(CONFIG_DIR, "camera_to_base_transformation.yaml"))

    # Load detection area
    detection_area = load_detection_area(os.path.join(CONFIG_DIR, "detection_area.yaml"))
    if detection_area:
        print(f"Detection area loaded: {detection_area}")
    else:
        print("No detection area configured. Detecting in entire image.")

    # Initialize camera thread
    camera_thread = CameraThread(MODEL_PATH, LABEL_MAP, COLOR_MAP, detection_area)
    camera_thread.start()

    print("Camera streaming started with OBB detection. The arm will automatically pick detected screws.")

    is_moving = False
    has_picked = False

    device.move_to(250, 0, 51, wait=True)
    device.move_to(250, 0, 50, wait=True)
    print("Dobot home completed.")

    try:
        while camera_thread.is_alive():
            print(f"Detections: {len(camera_thread.detection_results)}")
            if not is_moving and not has_picked and camera_thread.detection_results:
                # Find the highest confidence detection
                best_detection = max(camera_thread.detection_results, key=lambda d: d['conf'], default=None)
                if best_detection and best_detection['conf'] > 0.1:  # Lower threshold for testing
                    print(f"Best detection: {best_detection['label']} with confidence {best_detection['conf']:.2f}")
                    if best_detection['point_3d'] is not None:
                        point_coordinate = np.array(best_detection['point_3d']).reshape(3,1)
                        target_coordinate = R_camera2base @ point_coordinate + t_camera2base.reshape(3,1)
                        print(f"Target coordinate: {target_coordinate.flatten()}")
                        # Move to pick
                        is_moving = True
                        camera_thread.detecting = False  # Stop detection during movement
                        device.move_to(target_coordinate[0], target_coordinate[1], target_coordinate[2] + 50, wait=True)  # Above
                        device.clear_alarms()
                        device.move_to(target_coordinate[0], target_coordinate[1], target_coordinate[2] + 10, wait=True)  # Pick
                        device.clear_alarms()
                        device.grip(True)  # Close gripper
                        device.move_to(target_coordinate[0], target_coordinate[1], target_coordinate[2] + 50, wait=True)
                        device.clear_alarms()
                        device.grip(False)  # Open gripper to release
                        is_moving = False
                        print(f"Picked {best_detection['label']} at {target_coordinate.flatten()}")
                        # Return to home
                        device.move_to(250, 0, 50, wait=True)
                        camera_thread.detecting = True  # Resume detection
                        has_picked = True
                        print("Picking completed.")
                    else:
                        print("No valid 3D point for detection")

            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        camera_thread.stop()
        camera_thread.join()
        device.close()


if __name__ == "__main__":
    main()

