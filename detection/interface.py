"""Object detection interface using YOLOv8 OBB and Intel RealSense.

This module keeps only the object-detection interface. It initializes a
RealSense pipeline, loads a YOLO model (path from MODEL_PATH env or
local 'best.pt'), runs detection, and displays a simple Live preview.

This variant assumes the `pyrealsense2` and `ultralytics` packages are
installed and available at import time.
"""

from typing import List, Optional, Tuple
import os
import cv2
import numpy as np

import pyrealsense2 as rs
from ultralytics import YOLO

MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), '../demo.pt'))
MIN_CONFIDENCE_THRESHOLD = float(os.environ.get('MIN_CONFIDENCE_THRESHOLD', 0.5))

def main(model_path: str = MODEL_PATH, min_conf: float = MIN_CONFIDENCE_THRESHOLD):
    """Run a live preview using RealSense and YOLO.

    Assumes RealSense and YOLO are available. If model file is missing the
    function will print an error and return.
    """
    # Initialize RealSense pipeline, profile and alignment (inlined)
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    # Align depth to color
    align = rs.align(rs.stream.color)

    # Load model
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        pipeline.stop()
        return
    model = YOLO(model_path)

    print('Press Q to quit.')
    try:
        while True:
            # Inline of process_frames(pipeline, align)
            if pipeline is None or align is None:
                color_image, color_frame, depth_frame = None, None, None
            else:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    color_image, color_frame, depth_frame = None, None, None
                else:
                    color_image = np.asanyarray(color_frame.get_data())

            if color_image is None:
                continue

            results = model(color_image, verbose=False, conf=min_conf, save=False, stream=True)

            total_raw = 0
            valid = 0
            detections: List[dict] = []

            for r in results:
                obb = getattr(r, 'obb', None)
                if obb is None or not hasattr(obb, 'conf'):
                    continue
                total_raw = len(obb.conf)
                for i in range(len(obb.conf)):
                    conf = float(obb.conf[i])
                    if conf < min_conf:
                        continue
                    valid += 1
                    corners = np.array(obb.xyxyxyxy[i].cpu())
                    class_id = int(obb.cls[i])
                    detections.append({
                        'center': np.mean(corners, axis=0),
                        'corners': corners,
                        'confidence': conf,
                        'class_id': class_id,
                    })

            for detection in detections:
                corners = detection['corners'].astype(int)
                for j in range(4):
                    pt1 = tuple(corners[j])
                    pt2 = tuple(corners[(j + 1) % 4])
                    cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)
                label = f"{int(detection['class_id'])} ({detection['confidence']:.2f})"
                cv2.putText(color_image, label, tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            debug_text = f"Raw: {total_raw}, Valid: {valid}"
            cv2.putText(color_image, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('YOLOv11 OBB RealSense Preview', color_image)
            key = cv2.waitKey(1)
            if key & 0xFF in [ord('q'), ord('Q')]:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
