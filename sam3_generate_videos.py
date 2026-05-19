import os
import sam3
import torch

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
gpus_to_use = [3]

from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

from collections import defaultdict
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

# font size for axes titles
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12


def generate_distinct_palette(n=40):
    """
    Generate n highly distinguishable colors (BGR for OpenCV)
    using perceptually better distribution in HSV with golden ratio spacing.
    """

    colors = []

    # Golden ratio for good hue dispersion
    golden_ratio_conjugate = 0.61803398875
    h = np.random.rand()

    for i in range(n):
        h = (h + golden_ratio_conjugate) % 1.0

        hue = int(h * 179)  # OpenCV hue range

        # Alternate saturation and value in a more controlled way
        s = 200 + (i % 2) * 55      # 200–255
        v = 200 + ((i // 2) % 2) * 55

        hsv = np.uint8([[[hue, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]

        colors.append(tuple(int(c) for c in bgr))

    return colors



def min_bounding_box(arr):
    """
    Returns the bounding box of True values in a 2D boolean numpy array.
    Returns (min_row, min_col, max_row, max_col)
    """
    # Get the indices of True values
    true_indices = np.argwhere(arr)
    
    if true_indices.size == 0:
        # No True values
        return None
    
    # Get min and max along each axis
    min_row, min_col = true_indices.min(axis=0)
    max_row, max_col = true_indices.max(axis=0)
    
    return (min_row, min_col, max_row, max_col)


def group_bboxes_by_groupId(data, exclude_label="individual"):
    from collections import defaultdict
    group_map = defaultdict(list)

    for item in data:
        if item["label"] != exclude_label:
            group_map[item["groupId"]].append(item["bbox"])

    return dict(group_map)


def iou(boxA, boxB):
    # box: [x1, y1, x2, y2] OR (y1, x1, y2, x2) → normalize first
    xA1, yA1, xA2, yA2 = boxA
    yB1, xB1, yB2, xB2 = boxB  # second format

    # convert B to (x1, y1, x2, y2)
    xB1, yB1, xB2, yB2 = xB1, yB1, xB2, yB2

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def match_boxes(group_dict, track_dict, iou_thresh=0.6):
    matches = []
    used_tracks = set()

    for gid, bboxes in group_dict.items():
        for bbox in bboxes:
            best_match = None
            best_iou = 0

            for tid, tbox in track_dict.items():
                if tid in used_tracks:
                    continue

                score = iou(bbox, tbox)

                if score > iou_thresh and score > best_iou:
                    best_iou = score
                    best_match = tid

            if best_match is not None:
                matches.append((gid, best_match, best_iou))
                used_tracks.add(best_match)

    return matches

def visualize_tracks(image, track_dict):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(img)

    for tid, box in track_dict.items():
        y1, x1, y2, x2 = box  # convert format

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linestyle="dashed",
            linewidth=2
        )
        ax.add_patch(rect)

        ax.text(x1, y1 - 5, f"T{tid}", fontsize=8)

    plt.axis("off")
    plt.show()
    
def visualize_bboxes_on_image(image, group_dict):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for gid, boxes in group_dict.items():
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False
            )
            ax.add_patch(rect)
            ax.text(x1, y1, f"{gid}", fontsize=8)

    plt.axis("off")
    plt.show()

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False

        # union by rank
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

        return True


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")
        
files = glob.glob('SEKAI_900_3/videos_frames/*')
files.sort()

with open('results/finegrained_all_annotations.json', 'r') as f:
    finegrained_annotations = json.load(f)

video_annotations = defaultdict(lambda: defaultdict(list))    

for id_video, video_path in enumerate(files):
    print(id_video)    
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpeg"))
        try:
            # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()
    
        
    response = predictor.handle_request(request=dict(type="start_session", resource_path=video_path, ))
    session_id = response["session_id"]
    

    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )
    
    prompt_text_str = "person"
    frame_idx = 0  # add a text prompt on frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )
    out = response["outputs"]
    
    # plt.close("all")
    # visualize_formatted_frame_output(frame_idx,video_frames_for_vis,outputs_list=[prepare_masks_for_visualization({frame_idx: out})],titles=["SAM 3 Dense Tracking outputs"],figsize=(6, 4),)
    
    outputs_per_frame = propagate_in_video(predictor, session_id)
    #print(len(outputs_per_frame))
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
    #print(len(outputs_per_frame))

    #vis_frame_stride = 1
    #plt.close("all")
    
    #for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
    #    visualize_formatted_frame_output(frame_idx,video_frames_for_vis,outputs_list=[outputs_per_frame],titles=["SAM 3 Dense Tracking outputs"],figsize=(6, 4),)
    
    #video_annotations = defaultdict(list)

    track_to_group= {i:i for i in range(100)}
    print(video_path)
    videoIndex = int(video_path.split('_')[-1])
    print(videoIndex)
    matched_ann = [ann for ann in finegrained_annotations['annotations'] if ann['videoIndex'] == videoIndex] # CHANGE
    
    uf = UnionFind(10000)
    
    for ann in matched_ann:
        group_boxes = group_bboxes_by_groupId(ann['groups'])
        track_to_bbox = {tid: min_bounding_box(mask) for tid, mask in outputs_per_frame[ann['videoInfo']['annotationFrame']].items()}
        matches = match_boxes(group_boxes, track_to_bbox)
        file_path = f"{video_path}/{(str(ann['videoInfo']['annotationFrame']+1)).zfill(5)}.jpeg"
        I = cv2.imread(file_path)
        
        groups = defaultdict(list)
        for a, b, _ in matches:
            groups[a].append(b)
    
        for bs in groups.values():
            for b in bs[1:]:
                uf.union(bs[0], b)
    
    for fig_id, detection in enumerate(outputs_per_frame.values()):
        flag = False
        for track_id in detection:
            mask = detection[track_id]
            bbox = min_bounding_box(mask)
            annotation = {}
            annotation['bbox'] = [int(x) for x in [bbox[1], bbox[0], bbox[3], bbox[2]]]
            annotation['track_id'] = int(track_id)
            annotation['group_id'] = uf.parent[track_id]
            video_annotations[id_video+1][fig_id].append(annotation)
            flag = True
        
        if flag == False:
            video_annotations[id_video+1][fig_id].append({})

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))


    if id_video % 10 == 0:
        print('Saving - ID: ', id_video)
        with open('results/tracking_annotations.json', 'w') as f:
            json.dump(dict(video_annotations), f)

with open('results/tracking_annotations.json', 'w') as f:
    json.dump(dict(video_annotations), f)
        
