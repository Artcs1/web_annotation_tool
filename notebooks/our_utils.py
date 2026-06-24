import numpy as np
import math
import cv2
import torch

from PIL import Image

import os
import glob

def retrieve_video_paths(paths, dataset):

    full_video_paths = []

    if dataset == 'sekai':

        for path in paths:
            safe_dir = glob.escape(path)
            pattern = os.path.join(safe_dir, '*.mp4')
            files = glob.glob(pattern)
            files.sort()
            if files != []:
                tam = len(files)
                if tam > 1:
                    full_video_paths.extend(files)

    else:
        for path in paths:
            dirs = glob.glob(path)
            n_fl = len(dirs)
            cont = 0
            individual_files = 0
            for item_dir in dirs:
                safe_dir = glob.escape(item_dir)
                pattern = os.path.join(safe_dir, '*.mp4')
                files = glob.glob(pattern)
                files.sort()
                if files != []:
                    cont+=1
                    tam = len(files)
                    individual_files+=len(files)
                    if tam > 1:
                        full_video_paths.extend(files[1:])

    return full_video_paths


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def draw_tracking(frame, bbox, label, color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def pil_to_opencv(pil_img):
    """
    Convert a PIL Image to an OpenCV image (NumPy array).
    Handles RGB, RGBA, and L (grayscale).
    """
    # Ensure it's a PIL Image object
    if not isinstance(pil_img, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    # Convert to NumPy
    np_img = np.array(pil_img)

    # Handle different PIL modes
    if pil_img.mode == "RGB":
        # PIL (RGB) -> OpenCV (BGR)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    elif pil_img.mode == "RGBA":
        # PIL (RGBA) -> OpenCV (BGRA)
        return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)
    elif pil_img.mode == "L":
        # PIL (grayscale) just becomes 2D array, no color channel swap needed
        return np_img
    else:
        # Fallback: convert PIL to RGB first
        rgb_img = pil_img.convert("RGB")
        np_img = np.array(rgb_img)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def opencv_to_pil(cv_img):
    """
    Convert an OpenCV image (NumPy array, BGR/BGRA/Gray) back to a PIL Image.
    """
    if not isinstance(cv_img, np.ndarray):
        raise TypeError("Input must be a NumPy array (OpenCV image).")

    # Check shape to figure out color space
    if len(cv_img.shape) == 2:
        # Grayscale
        return Image.fromarray(cv_img)
    elif len(cv_img.shape) == 3:
        channels = cv_img.shape[2]
        if channels == 3:
            # OpenCV (BGR) -> PIL (RGB)
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        elif channels == 4:
            # OpenCV (BGRA) -> PIL (RGBA)
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA))

    # Fallback: Convert to RGB
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def vqa_yes_prob(model_7b, processor_7b, image, question_prompt, device="cuda"):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": 224 * 224,
                },
            {"type": "text", "text": f"{question_prompt}? Answer yes or no."},
        ],
        }
    ]

    text = processor_7b.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor_7b(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",)


    inputs = inputs.to(model_7b.device)

    with torch.no_grad():
        outputs = model_7b.generate(**inputs, max_new_tokens=128,return_dict_in_generate=True,use_cache=False,output_scores=True,do_sample=False)
        generated_ids = outputs["sequences"]
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor_7b.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        inputs_for_prob = processor_7b(text=["Yes"],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt")
        yes_id = inputs_for_prob['input_ids'][0][0].item()


        inputs_for_prob = processor_7b(text=["No"],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt")
        no_id = inputs_for_prob['input_ids'][0][0].item()

        logits = outputs["scores"][0][0]
        probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id]]), dim=0).detach().cpu().numpy())
            # print(f"Output text: {output_text} :: {rotation_ii} - Yes prob: {probs[0]}, No prob: {probs[1]}")
            
        return probs[0], output_text #return yes prob

def image_to_base64_str(img_array):
    # img_array is your NumPy array in BGR format from OpenCV
    # Encode as PNG in memory
    success, encoded_image = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Could not encode image as PNG.")
    # Convert to base64 bytes, then decode as ASCII/UTF-8 to get a string
    b64_str = base64.b64encode(encoded_image).decode('utf-8')
    return b64_str

def base64_str_to_image2(b64_str):
    # Remove header if present
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[-1]
    image_data = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(image_data))
    # Convert PIL image (RGB) to OpenCV format (BGR)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def get_intrinsics(H, W, fov=55):
    f = 0.5*W/np.tan(0.5*fov*np.pi/180.0)
    cx = 0.5 * H
    cy = 0.5 * W
    return np.array([[f,0,cx],
                     [0,f,cy],
                     [0,0,1]])

def pixel_to_point(depth_image, normalize, camera_intrinsics=None):
    
    height, width = depth_image.shape
    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(height, width, fov=55.0)

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)

    u, v = np.meshgrid(x, y)

    x_over_z = (u - cx) / (fx)
    y_over_z = (v - cy) / (fy)

    # 3-D Pythagoras re-arranged to solve for z
    if normalize:
        z = depth_image/ np.sqrt(1. + x_over_z**2 + y_over_z**2)
    else:
        z = depth_image
        
    x = x_over_z * z
    y = y_over_z * z

    return x, y, z

def plot_coord(coord, center, direction, offset, rgb_tuple):
    
    cv2.circle(coord, center, 10, color=rgb_tuple, thickness=-1)
    if direction == 'front':
        end_point = (center[0], center[1]+offset)
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)
    elif direction == 'front right':
        end_point = (center[0]-offset, center[1]+offset)
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)
    elif direction == 'front rright':
        end_point = (center[0]-offset, center[1])
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)
    elif direction == 'front left':
        end_point = (center[0]+offset, center[1]+offset)
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)
    elif direction == 'front lleft':
        end_point = (center[0]+offset, center[1])
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)
    elif direction == 'back':
        end_point = (center[0], center[1]-offset)
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)
    elif direction == 'back right':
        end_point = (center[0]+offset, center[1]-offset)
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)
    elif direction == 'back left':
        end_point = (center[0]-offset, center[1]-offset)
        cv2.arrowedLine(coord, center, end_point, color=(0, 0, 255), thickness=3, tipLength=0.5)

