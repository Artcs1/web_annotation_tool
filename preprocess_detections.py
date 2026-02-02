#!/usr/bin/env python3
"""
Preprocess detections for all videos
Runs YOLO on each video's first frame and saves results to cache
"""

import json
import cv2
from pathlib import Path
from ultralytics import YOLO

# Paths
VIDEOS_DIR = Path(__file__).parent / "videos"
COARSE_ANNOTATIONS_PATH = Path(__file__).parent / "coarse_group" / "all_annotations.json"
DETECTIONS_CACHE = Path(__file__).parent / "detections_cache.json"

# Load model
print("[INFO] Loading YOLOv8n model...")
yolo_model = YOLO("yolov8n.pt")

# Load coarse annotations
with open(COARSE_ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
    COARSE_ANNOTATIONS = json.load(f)

# Cache structure
detections_cache = {}

print(f"[INFO] Processing {len(COARSE_ANNOTATIONS['annotations'])} videos...")

for ann_idx, ann in enumerate(COARSE_ANNOTATIONS['annotations']):
    video_index = ann['videoIndex']
    ann_frame = ann['videoInfo']['annotationFrame']
    folder = ann.get('folder', f"clip_{video_index:04d}")
    # Load first frame

    frame_path = VIDEOS_DIR / folder / f"{ann_frame+1:05d}.jpeg"
    print(frame_path)
    
    if not frame_path.exists():
        print(f"[WARN] Frame not found: {frame_path}")
        continue
    
    print(f"[{ann_idx + 1}/{len(COARSE_ANNOTATIONS['annotations'])}] Processing {folder}...")
    
    image = cv2.imread(str(frame_path))
    if image is None:
        print(f"[ERROR] Failed to load image: {frame_path}")
        continue
    
    # Run YOLO detection
    results = yolo_model(image, conf=0.5, classes=0, verbose=False)
    
    # Extract detections
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(box.conf[0].cpu().numpy())
            })
    
    key = str(video_index)+'_'+str(ann_frame)
    print(key)
    detections_cache[key] = {
        'videoIndex': video_index,
        'folder': folder,
        'annotatedFrame': ann_frame,
        'detections': detections,
        'count': len(detections)
    }
    
    print(f"    → Found {len(detections)} persons")

# Save cache
with open(DETECTIONS_CACHE, 'w', encoding='utf-8') as f:
    json.dump(detections_cache, f, indent=2)

print(f"\n[INFO] Detections saved to: {DETECTIONS_CACHE}")
print(f"[INFO] Total detections cached: {sum(v['count'] for v in detections_cache.values())}")
