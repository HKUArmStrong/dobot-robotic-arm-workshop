
"""Quick test runner for a YOLO model.

Usage examples:
  # point directly at an image
  python detection/test_yolo.py --image dataset/screw.v2i.yolov8-obb/train/images/IMG_...jpg

  # pick the first image from the dataset directory and display/save the result
  python detection/test_yolo.py --dataset dataset/screw.v2i.yolov8-obb --show

The script supports both OBB-style and standard bbox-style models/results.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---------------------------------
# Optional: set the image filename you want to use from the raw_image directory.
# If left empty, the script will pick the first image in the directory.
# You can also override at runtime with the IMAGE_NAME environment variable.
IMAGE_NAME = os.environ.get('IMAGE_NAME', 'IMG_09.jpg')  # e.g. 'IMG_01.jpg' or '' to auto-pick

# Directory containing raw images. Override with RAW_IMAGE_DIR environment variable.
# Default mirrors the project layout: ../raw_image
RAW_IMAGE_DIR = os.environ.get('RAW_IMAGE_DIR', os.path.join(os.path.dirname(__file__), '../raw_image'))
# ---------------------------------------------------------------------

# Model path macro (set here or override with MODEL_PATH env var)
# MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), '../detection/model/demo.pt'))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), '../detection/model/yolo11m-screw/weights/best.pt'))


def to_numpy(x):
	# Convert tensors or other array-like objects to numpy arrays safely
	try:
		if hasattr(x, 'cpu'):
			return x.cpu().numpy()
		if hasattr(x, 'numpy'):
			return x.numpy()
	except Exception:
		pass
	return np.array(x)

def draw_obb(img, corners, color=(0, 255, 0), thickness=2):
	# corners expected shape (4,2)
	pts = corners.astype(int)
	for i in range(4):
		pt1 = tuple(pts[i])
		pt2 = tuple(pts[(i + 1) % 4])
		cv2.line(img, pt1, pt2, color, thickness)


def draw_bbox(img, xyxy, color=(255, 0, 0), thickness=2):
	x1, y1, x2, y2 = map(int, xyxy)
	cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--image', type=str, help='Path to an image file to run (optional)')
	p.add_argument('--min-conf', type=float, default=0.25, help='Minimum confidence to display')
	p.add_argument('--show', action='store_true', help='Show image in a window (do not save)')
	args = p.parse_args()

	# determine image: use --image if provided, otherwise use IMAGE_NAME (macro),
	# otherwise pick first image from RAW_IMAGE_DIR
	img_path = None
	if args.image:
		img_path = Path(args.image)
		if not img_path.is_file():
			raise FileNotFoundError(f"Image not found: {img_path}")
	else:
		raw_dir = Path(RAW_IMAGE_DIR).resolve()
		if not raw_dir.is_dir():
			raise FileNotFoundError(f"Raw image directory not found: {raw_dir}")

		# If IMAGE_NAME is set (or IMAGE_NAME env was provided), use that file
		if IMAGE_NAME:
			candidate = raw_dir / IMAGE_NAME
			if candidate.is_file():
				img_path = candidate
			else:
				raise FileNotFoundError(f"IMAGE_NAME set but file not found: {candidate}")

		# Otherwise pick the first supported image file in raw_dir
		if img_path is None:
			files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
			if files:
				img_path = files[0]

	if img_path is None:
		raise ValueError('No image found. Provide --image or place images in RAW_IMAGE_DIR (or set IMAGE_NAME).')

	print(f"Using image: {img_path}")

	img = cv2.imread(str(img_path))
	if img is None:
		raise RuntimeError(f"Failed to read image {img_path}")

	# load model (use MODEL_PATH macro)
	model_path = MODEL_PATH
	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"Model file not found: {model_path}\nSet MODEL_PATH env or edit the MODEL_PATH constant in the script.")
	print(f"Loading model: {model_path}")
	model = YOLO(model_path)

	# run inference
	results = model(img, conf=args.min_conf, verbose=False)

	annotated = img.copy()
	total = 0
	kept = 0

	for r in results:
		# OBB-style
		obb = getattr(r, 'obb', None)
		if obb is not None and hasattr(obb, 'conf'):
			total = len(obb.conf)
			for i in range(total):
				conf = float(obb.conf[i])
				if conf < args.min_conf:
					continue
				kept += 1
				# read corners; handle tensor or array
				corners = to_numpy(obb.xyxyxyxy[i])
				# corners may flatten to 8 values
				corners = corners.reshape((4, 2))
				draw_obb(annotated, corners)
				# label
				cls = int(to_numpy(obb.cls[i]))
				label = f"{cls}, {conf:.2f}"
				cv2.putText(annotated, label, tuple(corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
			continue

		# standard bbox style
		boxes = getattr(r, 'boxes', None)
		if boxes is not None and hasattr(boxes, 'xyxy'):
			xyxy = to_numpy(boxes.xyxy)
			confs = to_numpy(boxes.conf)
			clss = to_numpy(boxes.cls)
			total = len(confs)
			for i in range(total):
				conf = float(confs[i])
				if conf < args.min_conf:
					continue
				kept += 1
				box = xyxy[i]
				draw_bbox(annotated, box)
				label = f"{int(clss[i])},{conf:.2f}"
				cv2.putText(annotated, label, (int(box[0]), int(box[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

	print(f"Detections: raw={total}, shown={kept}")

	# show or save output
	if args.show:
		# Show the annotated image on screen only (do not save)
		cv2.imshow('Annotated', annotated)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		out_path = img_path.with_name(img_path.stem + '_annotated' + img_path.suffix)
		out_path = Path(out_path)
		cv2.imwrite(str(out_path), annotated)
		print(f"Annotated image saved to: {out_path}")


if __name__ == '__main__':
	main()

