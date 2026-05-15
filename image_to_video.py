import cv2
import os

def images_to_video(image_folder, output_path, fps=30):
    # Get all image files and sort them
    images = [img for img in os.listdir(image_folder) 
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    if not images:
        raise ValueError("No images found in the folder.")

    # Read first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Skipping unreadable image: {image}")
            continue

        # Resize if needed (ensures consistent size)
        frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    print(f"Video saved at: {output_path}")


# Example usage
images_to_video(
    image_folder="SEKAI_720_3/videos_frames/clip_0056/",
    output_path="output.mp4",
    fps=30
)
