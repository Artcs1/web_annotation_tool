"""
SAM3 Person Detection - Following V6 Two-Stage Architecture Exactly
Stage 1: Coarse patches (crowded groups) with lower confidence
Stage 2: Full image (scattered people) with higher confidence
Dedup: IoU-based with source tracking
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# SAM3 Models
MODEL_DIR = Path("model/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT = MODEL_DIR / "sam3" / "sam3.pt"

print(MODEL_CHECKPOINT)

# Import SAM3
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[INFO] SAM3 imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import SAM3: {e}")
    print("[INFO] Make sure SAM3 is installed: pip install -e /path/to/sam3")
    exit(1)

# Paths
BASE_DIR = Path(__file__).parent
ANNOTATIONS_FILE = BASE_DIR / "coarse_group" / "all_annotations.json"
VIDEOS_DIR = BASE_DIR / "videos"
CACHE_FILE = BASE_DIR / "detections_cache_sam3.json"

def iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]"""
    # Squeeze out batch dimension if present (SAM3 returns [1, H, W])
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    
    # For 2D mask, np.where returns (row_indices, col_indices) = (y, x)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

def detect_persons_sam3(image_pil, processor, inference_state, confidence_threshold=0.65):
    """Run SAM3 person detection on image using text prompt"""
    try:
        # Set image
        inference_state = processor.set_image(image_pil)
        
        # Prompt with text: "person"
        output = processor.set_text_prompt(
            state=inference_state, 
            prompt="person"
        )
        
        masks = output.get("masks", [])  # Shape: [num_masks, height, width]
        scores = output.get("scores", [])  # Confidence scores
        
        detections = []
        
        if len(masks) > 0:
            masks = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
            scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            
            for mask, score in zip(masks, scores):
                if score < confidence_threshold:
                    continue
                    
                # Convert mask to bbox
                bbox = mask_to_bbox(mask)
                if bbox is None:
                    continue
                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(score),
                    'mask': mask.astype(np.uint8)  # Store mask for visualization
                })
        
        return detections, inference_state
        
    except Exception as e:
        print(f"[ERROR] SAM3 detection failed: {e}")
        return [], inference_state

def dedup_detections(all_detections, iou_threshold=0.5):
    """Remove duplicate detections using NMS"""
    if not all_detections:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
    
    # NMS
    keep = []
    for det in sorted_dets:
        is_duplicate = False
        for kept_det in keep:
            if iou(det['bbox'], kept_det['bbox']) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(det)
    
    return keep

def process_videos():
    """Process videos and extract person detections (V6 two-stage logic)"""

    print("[INFO] Loading SAM3 model...")
    try:
        if MODEL_CHECKPOINT.exists():
            print(MODEL_CHECKPOINT)
            model = build_sam3_image_model(
                checkpoint_path=str(MODEL_CHECKPOINT), load_from_HF=False
            )
            print(str(MODEL_CHECKPOINT))
        else:
            print(
                f"[WARNING] Checkpoint not found at {MODEL_CHECKPOINT}. Downloading from HuggingFace..."
            )
            model = build_sam3_image_model()
        processor = Sam3Processor(model)
        print("[INFO] SAM3 model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load SAM3 model: {e}")
        print("[INFO] This usually means the model checkpoint hasn't been downloaded yet.")
        print("[INFO] The model will be downloaded automatically on first use.")
        return {}

    # Load annotations
    print(f"[INFO] Loading coarse annotations from {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE) as f:
        annotations_data = json.load(f)

    detections_cache = {}
    coarse_patch_detections = 0
    full_image_detections = 0

    print(f"\n[INFO] Processing {len(annotations_data['annotations'])} annotation frames...")
    print("[INFO] Using V6 Two-Stage Architecture:")
    print("       Stage 1: Coarse patches sorted by area (small→large, conf=0.70)")
    print("       Stage 2: Full image (conf=0.65)")
    print("       Dedup: IoU threshold=0.50 (balanced)")
    print()

    for annotation in tqdm(annotations_data["annotations"], desc="Processing frames"):
        video_index = annotation["videoIndex"]
        frame_index = annotation["videoInfo"]["annotationFrame"]
        coarse_groups = annotation["groups"]

        # Load frame
        frame_path = VIDEOS_DIR / f"clip_{video_index:04d}" / f"{frame_index + 1:05d}.jpeg"
        if not frame_path.exists():
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        all_detections = []
        inference_state = None

        # ========== STAGE 1: Coarse patch detection ==========
        if len(coarse_groups) > 0:
            # Sort groups by area (small to large) for better detection in crowded areas
            sorted_groups = sorted(
                coarse_groups,
                key=lambda g: (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1])
            )
            
            for group in sorted_groups:
                gx1, gy1, gx2, gy2 = [int(x) for x in group["bbox"]]
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w, gx2), min(h, gy2)

                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                patch = image_rgb[gy1:gy2, gx1:gx2]
                if patch.size == 0:
                    continue

                patch_pil = Image.fromarray(patch)

                # Detect on patch with confidence 0.70 for quality
                dets, inference_state = detect_persons_sam3(
                    patch_pil, processor, inference_state, confidence_threshold=0.70
                )

                # Adjust bbox to full image coordinates
                for det in dets:
                    det["bbox"] = [
                        det["bbox"][0] + gx1,
                        det["bbox"][1] + gy1,
                        det["bbox"][2] + gx1,
                        det["bbox"][3] + gy1,
                    ]
                    det["source"] = "coarse_patch"
                    all_detections.append(det)
                    coarse_patch_detections += 1

        # ========== STAGE 2: Full image detection ==========
        image_pil = Image.fromarray(image_rgb)
        full_dets, inference_state = detect_persons_sam3(
            image_pil, processor, inference_state, confidence_threshold=0.65
        )

        for det in full_dets:
            det["source"] = "full_image"
            all_detections.append(det)
            full_image_detections += 1

        # ========== DEDUPLICATION ==========
        if len(all_detections) > 1:
            all_detections.sort(key=lambda x: x["confidence"], reverse=True)

            final_detections = []
            for det in all_detections:
                is_duplicate = False

                for kept_det in final_detections:
                    overlap_iou = iou(det["bbox"], kept_det["bbox"])

                    if overlap_iou > 0.50:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    final_detections.append(det)
        else:
            final_detections = all_detections

        # Store in cache (remove mask for storage)
        cache_key = f"{video_index}_{frame_index}"
        detections_cache[cache_key] = {
            "count": len(final_detections),
            "detections": [
                {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "source": det["source"],
                }
                for det in final_detections
            ],
        }

    # Save cache
    print(f"\n[INFO] Saving cache to {CACHE_FILE}")
    with open(CACHE_FILE, "w") as f:
        json.dump(detections_cache, f)

    # Statistics
    total_detections = sum(d["count"] for d in detections_cache.values())
    avg_per_frame = total_detections / len(detections_cache) if detections_cache else 0

    print("\n" + "=" * 70)
    print("SAM3 RESULTS (V6 Two-Stage: Coarse 0.70 / Full 0.65, IoU=0.50)")
    print("=" * 70)
    print(f"Total frames processed: {len(detections_cache)}")
    print(f"Coarse patch detections: {coarse_patch_detections}")
    print(f"Full image detections: {full_image_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Average per frame: {avg_per_frame:.2f}")
    print("=" * 70)

    return detections_cache

def save_detections(detections):
    """Save detections to cache file"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"[INFO] Saved {len(detections)} clips to {CACHE_FILE}")

def load_detections():
    """Load detections from cache file"""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    print("[INFO] Starting SAM3-based person detection...")
    
    # Always run to mirror V6 pipeline
    detections = process_videos()
    save_detections(detections)
    print(f"[INFO] Detection complete! Processed {len(detections)} frames")
