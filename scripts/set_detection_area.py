import cv2
import numpy as np
import pyrealsense2 as rs
import os
import yaml

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["points"].append((x, y))
        print(f"Point {len(param['points'])}: ({x}, {y})")
        if len(param["points"]) == 4:
            print("All four points selected. Press 's' to save or 'r' to reset.")

def initialize_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align

def main():
    pipeline, profile, align = initialize_pipeline()

    param = {"points": []}
    cv2.namedWindow("Set Detection Area")
    cv2.setMouseCallback("Set Detection Area", click_event, param)

    print("Click four corners of the detection area in order (e.g., top-left, top-right, bottom-right, bottom-left). Press 'q' to quit without saving.")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        display_image = color_image.copy()

        # Draw selected points
        for i, point in enumerate(param["points"]):
            cv2.circle(display_image, point, 5, (0, 255, 0), -1)
            cv2.putText(display_image, str(i+1), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw lines if 4 points
        if len(param["points"]) == 4:
            pts = np.array(param["points"], np.int32)
            cv2.polylines(display_image, [pts], True, (255, 0, 0), 2)

        cv2.imshow("Set Detection Area", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s") and len(param["points"]) == 4:
            # Save to config
            config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, "detection_area.yaml")
            with open(config_file, "w") as file:
                yaml.safe_dump({"detection_area": param["points"]}, file)
            print(f"Detection area saved to {config_file}")
            break
        elif key == ord("r"):
            param["points"] = []
            print("Points reset. Click four corners again.")

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()