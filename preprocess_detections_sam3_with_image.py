"""
SAM3 Person Detection - Following V6 Two-Stage Architecture Exactly
Stage 1: Coarse patches (crowded groups) with lower confidence
Stage 2: Full image (scattered people) with higher confidence
Dedup: IoU-based with source tracking

ADDED: Visualization of patch detections (kept vs discarded)
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
ANNOTATIONS_FILE = BASE_DIR / "results" / "all_annotations.json"
VIDEOS_DIR = BASE_DIR / "SEKAI_540_3"/ "videos_frames"
CACHE_FILE = BASE_DIR / "detections_cache_sam3.json"
VIZ_DIR = BASE_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

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

def visualize_patch_detections(image_rgb, orig_patch_bounds, expanded_patch_bounds, 
                               kept_detections, discarded_detections, 
                               frame_index, patch_idx, save_path=None):
    """
    Visualize patch detections showing kept vs discarded bounding boxes
    
    Args:
        image_rgb: Full image in RGB
        orig_patch_bounds: (x1, y1, x2, y2) - original patch boundary
        expanded_patch_bounds: (x1, y1, x2, y2) - expanded patch boundary
        kept_detections: List of [x1, y1, x2, y2, confidence] for kept boxes
        discarded_detections: List of [x1, y1, x2, y2, confidence] for discarded boxes
        frame_index: Frame number
        patch_idx: Patch index within the frame
        save_path: Optional path to save the visualization
    """
    vis_image = image_rgb.copy()
    orig_gx1, orig_gy1, orig_gx2, orig_gy2 = orig_patch_bounds
    expanded_gx1, expanded_gy1, expanded_gx2, expanded_gy2 = expanded_patch_bounds
    
    # Draw expanded patch boundary (orange)
    cv2.rectangle(vis_image, (expanded_gx1, expanded_gy1), (expanded_gx2, expanded_gy2), 
                 (255, 165, 0), 3)
    
    # Draw original patch boundary (green, thicker)
    cv2.rectangle(vis_image, (orig_gx1, orig_gy1), (orig_gx2, orig_gy2), 
                 (0, 255, 0), 4)
    
    # Draw kept detections (green boxes)
    for bbox in kept_detections:
        x1, y1, x2, y2, conf = bbox
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 2)
        # Draw corners to show which corner is inside
        corners = [(int(x1), int(y1)), (int(x2), int(y1)), 
                  (int(x1), int(y2)), (int(x2), int(y2))]
        for cx, cy in corners:
            if orig_gx1 <= cx < orig_gx2 and orig_gy1 <= cy < orig_gy2:
                cv2.circle(vis_image, (cx, cy), 5, (0, 255, 0), -1)
        
        cv2.putText(vis_image, f'KEPT {conf:.2f}', 
                   (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw discarded detections (red boxes)
    for bbox in discarded_detections:
        x1, y1, x2, y2, conf = bbox
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 0, 255), 2)
        cv2.putText(vis_image, f'DISCARD {conf:.2f}', 
                   (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add legend
    legend_y = 30
    legend_x = 10
    cv2.rectangle(vis_image, (legend_x, legend_y - 5), 
                 (legend_x + 250, legend_y + 135), (0, 0, 0), -1)  # Black background
    cv2.rectangle(vis_image, (legend_x, legend_y - 5), 
                 (legend_x + 250, legend_y + 135), (255, 255, 255), 2)  # White border
    
    cv2.putText(vis_image, 'Legend:', (legend_x + 5, legend_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.rectangle(vis_image, (legend_x + 5, legend_y + 25), (legend_x + 25, legend_y + 40), 
                 (255, 165, 0), -1)
    cv2.putText(vis_image, 'Expanded Patch', (legend_x + 30, legend_y + 38), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.rectangle(vis_image, (legend_x + 5, legend_y + 45), (legend_x + 25, legend_y + 60), 
                 (0, 255, 0), -1)
    cv2.putText(vis_image, 'Original Patch', (legend_x + 30, legend_y + 58), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.rectangle(vis_image, (legend_x + 5, legend_y + 65), (legend_x + 25, legend_y + 80), 
                 (0, 255, 0), 2)
    cv2.putText(vis_image, f'Kept ({len(kept_detections)})', (legend_x + 30, legend_y + 78), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.rectangle(vis_image, (legend_x + 5, legend_y + 85), (legend_x + 25, legend_y + 100), 
                 (0, 0, 255), 2)
    cv2.putText(vis_image, f'Discarded ({len(discarded_detections)})', (legend_x + 30, legend_y + 98), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.circle(vis_image, (legend_x + 15, legend_y + 115), 5, (0, 255, 0), -1)
    cv2.putText(vis_image, 'Corner in patch', (legend_x + 30, legend_y + 118), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Add title
    title = f'Frame {frame_index} - Patch {patch_idx}'
    cv2.putText(vis_image, title, (image_rgb.shape[1] - 300, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, title, (image_rgb.shape[1] - 300, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Save if path provided
    if save_path:
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), vis_image_bgr)
        print(f"[INFO] Saved visualization to {save_path}")
    
    # Show visualization
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow(f'Patch Detections - Frame {frame_index} Patch {patch_idx}', vis_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def visualize_full_image_results(image_rgb, final_detections, removed_detections, 
                                coarse_groups, frame_index, save_path=None):
    """
    Visualize full image detection results showing final detections and removed duplicates
    
    Args:
        image_rgb: Full image in RGB
        final_detections: List of final detection dicts with bbox, confidence, source
        removed_detections: List of removed detection dicts with bbox, confidence, source, removed_by
        coarse_groups: Original coarse group annotations
        frame_index: Frame number
        save_path: Optional path to save the visualization
    """
    vis_image = image_rgb.copy()
    
    # Draw original coarse group boundaries (light gray, dashed effect)
    for group in coarse_groups:
        gx1, gy1, gx2, gy2 = [int(x) for x in group["bbox"]]
        # Create dashed line effect
        dash_length = 10
        # Horizontal lines
        for x in range(gx1, gx2, dash_length * 2):
            cv2.line(vis_image, (x, gy1), (min(x + dash_length, gx2), gy1), (128, 128, 128), 2)
            cv2.line(vis_image, (x, gy2), (min(x + dash_length, gx2), gy2), (128, 128, 128), 2)
        # Vertical lines
        for y in range(gy1, gy2, dash_length * 2):
            cv2.line(vis_image, (gx1, y), (gx1, min(y + dash_length, gy2)), (128, 128, 128), 2)
            cv2.line(vis_image, (gx2, y), (gx2, min(y + dash_length, gy2)), (128, 128, 128), 2)
    
    # Draw removed/duplicate detections (semi-transparent red with X)
    for det in removed_detections:
        x1, y1, x2, y2 = [int(x) for x in det["bbox"]]
        # Draw thin red box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 100, 100), 1)
        # Draw X through the box
        cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(vis_image, (x2, y1), (x1, y2), (255, 0, 0), 2)
        # Label with source
        source_label = "P" if det["source"] == "coarse_patch" else "F"
        cv2.putText(vis_image, f'{source_label} {det["confidence"]:.2f}', 
                   (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Count detections by source
    patch_count = sum(1 for d in final_detections if d["source"] == "coarse_patch")
    full_count = sum(1 for d in final_detections if d["source"] == "full_image")
    
    # Draw final detections with different colors by source
    for det in final_detections:
        x1, y1, x2, y2 = [int(x) for x in det["bbox"]]
        
        if det["source"] == "coarse_patch":
            color = (0, 255, 0)  # Green for patch detections
            label = f'PATCH {det["confidence"]:.2f}'
        else:
            color = (0, 255, 255)  # Cyan for full image detections
            label = f'FULL {det["confidence"]:.2f}'
        
        # Draw thick box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
        
        # Add label with background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(vis_image, label, (x1 + 2, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add comprehensive legend
    legend_y = 30
    legend_x = 10
    legend_height = 220
    cv2.rectangle(vis_image, (legend_x, legend_y - 5), 
                 (legend_x + 280, legend_y + legend_height), (0, 0, 0), -1)
    cv2.rectangle(vis_image, (legend_x, legend_y - 5), 
                 (legend_x + 280, legend_y + legend_height), (255, 255, 255), 2)
    
    cv2.putText(vis_image, 'Full Image Results:', (legend_x + 5, legend_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Statistics
    y_offset = legend_y + 40
    cv2.putText(vis_image, f'Final Detections: {len(final_detections)}', 
               (legend_x + 5, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(vis_image, f'  From Patches: {patch_count}', 
               (legend_x + 5, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    y_offset += 18
    cv2.putText(vis_image, f'  From Full Image: {full_count}', 
               (legend_x + 5, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y_offset += 20
    cv2.putText(vis_image, f'Removed (Dedup): {len(removed_detections)}', 
               (legend_x + 5, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    
    # Legend items
    y_offset += 30
    cv2.rectangle(vis_image, (legend_x + 5, y_offset), (legend_x + 25, y_offset + 15), 
                 (128, 128, 128), 1)
    cv2.putText(vis_image, 'Coarse Groups', (legend_x + 30, y_offset + 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    y_offset += 22
    cv2.rectangle(vis_image, (legend_x + 5, y_offset), (legend_x + 25, y_offset + 15), 
                 (0, 255, 0), 3)
    cv2.putText(vis_image, 'Patch Detection', (legend_x + 30, y_offset + 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    y_offset += 22
    cv2.rectangle(vis_image, (legend_x + 5, y_offset), (legend_x + 25, y_offset + 15), 
                 (0, 255, 255), 3)
    cv2.putText(vis_image, 'Full Img Detection', (legend_x + 30, y_offset + 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    y_offset += 22
    cv2.rectangle(vis_image, (legend_x + 5, y_offset), (legend_x + 25, y_offset + 15), 
                 (255, 100, 100), 1)
    cv2.line(vis_image, (legend_x + 5, y_offset), (legend_x + 25, y_offset + 15), (255, 0, 0), 2)
    cv2.line(vis_image, (legend_x + 25, y_offset), (legend_x + 5, y_offset + 15), (255, 0, 0), 2)
    cv2.putText(vis_image, 'Removed (IoU>0.5)', (legend_x + 30, y_offset + 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Add title
    title = f'Frame {frame_index} - Final Results (2-Stage + Dedup)'
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    title_x = image_rgb.shape[1] - title_size[0] - 20
    cv2.putText(vis_image, title, (title_x, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(vis_image, title, (title_x, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save if path provided
    if save_path:
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), vis_image_bgr)
        print(f"[INFO] Saved full image visualization to {save_path}")
    
    # Show visualization
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow(f'Full Image Results - Frame {frame_index}', vis_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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
        #model = model.to(DEVICE).float()
        processor = Sam3Processor(model)
        #processor = Sam3Processor(model)
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

    for ann_id, annotation in tqdm(enumerate(annotations_data["annotations"]), desc="Processing frames"):

    #    if annotation['videoFolder'] != 'videos/clip_0077/':
    #        continue

    #    if annotation['videoInfo']['annotationFrame'] != 41:
    #        continue

        video_index = annotation["videoIndex"]
        frame_index = annotation["videoInfo"]["annotationFrame"]
        coarse_groups = annotation["groups"]

        # Load frame
        frame_path = VIDEOS_DIR / f"clip_{video_index:04d}" / f"{frame_index + 1:05d}.jpeg"
        print(frame_path)
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

            for patch_idx, group in enumerate(sorted_groups):
                gx1, gy1, gx2, gy2 = [int(x) for x in group["bbox"]]
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w, gx2), min(h, gy2)
                if gx2 <= gx1 or gy2 <= gy1:
                    continue
            
                # Store original patch boundaries
                orig_gx1, orig_gy1 = gx1, gy1
                orig_gx2, orig_gy2 = gx2, gy2
            
                # Expand by 1/3 in each direction
                patch_w = gx2 - gx1
                patch_h = gy2 - gy1
                expand_w = patch_w // 3
                expand_h = patch_h // 3
            
                # Apply expansion and clamp to image bounds
                expanded_gx1 = max(0, gx1 - expand_w)
                expanded_gy1 = max(0, gy1 - expand_h)
                expanded_gx2 = min(w, gx2 + expand_w)
                expanded_gy2 = min(h, gy2 + expand_h)
            
                # Extract expanded patch
                patch = image_rgb[expanded_gy1:expanded_gy2, expanded_gx1:expanded_gx2]
                if patch.size == 0:
                    continue
                patch_pil = Image.fromarray(patch)
            
                # Detect on expanded patch
                dets, inference_state = detect_persons_sam3(
                    patch_pil, processor, inference_state, confidence_threshold=0.70
                )
            
                # Track kept and discarded detections for visualization
                kept_detections = []
                discarded_detections = []
            
                # Adjust bbox to full image coordinates and filter
                for det in dets:
                    # Convert to full image coordinates
                    det_x1 = det["bbox"][0] + expanded_gx1
                    det_y1 = det["bbox"][1] + expanded_gy1
                    det_x2 = det["bbox"][2] + expanded_gx1
                    det_y2 = det["bbox"][3] + expanded_gy1
            
                    # Check if any corner is inside original patch

                    #corners = [
                    #    (det_x1, det_y1),  # top-left
                    #    (det_x2, det_y1),  # top-right
                    #    (det_x1, det_y2),  # bottom-left
                    #    (det_x2, det_y2),  # bottom-right
                    #]
                    corners = [
                        ((det_x1+det_x2)//2, (det_y1+det_y2)//2),  # center
                    ]
            
                    has_corner_inside = any(
                        orig_gx1 <= cx < orig_gx2 and orig_gy1 <= cy < orig_gy2
                        for cx, cy in corners
                    )
            
                    if has_corner_inside:
                        det["bbox"] = [det_x1, det_y1, det_x2, det_y2]
                        det["source"] = "coarse_patch"
                        all_detections.append(det)
                        kept_detections.append([det_x1, det_y1, det_x2, det_y2, det["confidence"]])
                        coarse_patch_detections += 1
                    else:
                        discarded_detections.append([det_x1, det_y1, det_x2, det_y2, det["confidence"]])
                
                # Visualize this patch's detections
                print(f"\n[DEBUG] Patch {patch_idx} detection summary:")
                print(f"  - Kept: {len(kept_detections)}")
                print(f"  - Discarded: {len(discarded_detections)}")
                
                save_path = VIZ_DIR / f"frame_{ann_id}_{frame_index}_patch_{patch_idx}.png"
                visualize_patch_detections(
                    image_rgb,
                    (orig_gx1, orig_gy1, orig_gx2, orig_gy2),
                    (expanded_gx1, expanded_gy1, expanded_gx2, expanded_gy2),
                    kept_detections,
                    discarded_detections,
                    frame_index,
                    patch_idx,
                    save_path=save_path
                )

        # ========== STAGE 2: Full image detection ==========
        image_pil = Image.fromarray(image_rgb)
        full_dets, inference_state = detect_persons_sam3(
            image_pil, processor, inference_state, confidence_threshold=0.65
        )

        # Track full image detections before dedup
        full_image_dets_list = []
        for det in full_dets:
            det["source"] = "full_image"
            all_detections.append(det)
            full_image_dets_list.append([det["bbox"][0], det["bbox"][1], 
                                         det["bbox"][2], det["bbox"][3], 
                                         det["confidence"]])
            full_image_detections += 1

        # ========== DEDUPLICATION ==========
        # Track which detections are removed during dedup
        removed_detections = []
        
        if len(all_detections) > 1:
            all_detections.sort(key=lambda x: x["confidence"], reverse=True)

            final_detections = []
            for det in all_detections:
                is_duplicate = False

                for kept_det in final_detections:
                    overlap_iou = iou(det["bbox"], kept_det["bbox"])

                    if overlap_iou > 0.50:
                        is_duplicate = True
                        removed_detections.append({
                            "bbox": det["bbox"],
                            "confidence": det["confidence"],
                            "source": det["source"],
                            "removed_by": kept_det["source"]
                        })
                        break

                if not is_duplicate:
                    final_detections.append(det)
        else:
            final_detections = all_detections
        
        # ========== VISUALIZE FULL IMAGE RESULTS ==========
        print(f"\n[DEBUG] Full image detection summary:")
        print(f"  - Stage 1 (patches): {coarse_patch_detections}")
        print(f"  - Stage 2 (full image): {len(full_image_dets_list)}")
        print(f"  - Total before dedup: {len(all_detections)}")
        print(f"  - Removed by dedup: {len(removed_detections)}")
        print(f"  - Final detections: {len(final_detections)}")
        
        save_path = VIZ_DIR / f"frame_{ann_id}_{frame_index}_full_image_final.png"
        visualize_full_image_results(
            image_rgb,
            final_detections,
            removed_detections,
            coarse_groups,
            frame_index,
            save_path=save_path
        )

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
    print("[INFO] Starting SAM3-based person detection with visualization...")
    
    # Always run to mirror V6 pipeline
    detections = process_videos()
    save_detections(detections)
    print(f"[INFO] Detection complete! Processed {len(detections)} frames")
    print(f"[INFO] Visualizations saved to {VIZ_DIR}")
