import numpy as np
import cv2

def preprocess_frame(frame):
    """
    frame: np.ndarray of shape (H, W, C)
    Removes:
      - bottom 1/5 (102 px from 512)
      - left 1/7 (42 px from 288)
    """
    height, width, _ = frame.shape

    bottom_crop = height // 5      # 102
    left_crop = width // 7         # 42

    # Crop: [top:bottom, left:right]
    cropped = frame[:height - bottom_crop, left_crop:, :]

    return cropped


def remove_background(frame, bg_color=(80, 190, 190), tolerance=30):
    """
    frame: RGB image (H, W, 3)
    bg_color: approximate background RGB
    tolerance: color distance threshold
    """
    frame = frame.copy()

    # Compute color distance from background
    diff = np.linalg.norm(frame - np.array(bg_color), axis=2)

    # Mask background
    background_mask = diff < tolerance

    # Set background to black
    frame[background_mask] = [0, 0, 0]

    return frame


def to_grayscale_with_white_borders(frame, border_mask):
    # Normal grayscale
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

    # Force pipe borders to white
    gray[border_mask] = 255

    return gray.astype(np.uint8)


def to_grayscale(frame):
    """
    frame: RGB image
    returns: grayscale image (H, W)
    """
    return np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def downscale_by_3(gray_frame):
    """
    gray_frame: (H, W) uint8 grayscale image
    returns: downscaled image by factor of 3
    """
    h, w = gray_frame.shape
    new_size = (w // 3, h // 3)

    return cv2.resize(
        gray_frame,
        new_size,
        interpolation=cv2.INTER_AREA
    )


def pipe_border_mask(r, g, b):
    return (
        (r > 40) & (r < 100) &
        (g > 10) & (g < 60) &
        (b > 15) & (b < 80)
    )



def bird_mask(r, g, b):
    yellow = (r > 220) & (g > 180) & (b < 90)
    red    = (r > 200) & (g < 150) & (b < 80)
    white  = (r > 230) & (g > 230) & (b > 230)
    black  = (r < 50)  & (g < 50)  & (b < 50)  # eye outline

    return yellow | red | white | black


import numpy as np

def keep_bird_and_pipe_borders(frame):
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]

    border = pipe_border_mask(r, g, b)
    bird   = bird_mask(r, g, b)

    keep = border | bird

    out = frame.copy()
    out[~keep] = [0, 0, 0]

    return out, border


import os
import cv2
import numpy as np

def save_debug_image(img, name, out_dir="debug"):
    os.makedirs(out_dir, exist_ok=True)

    # Convert float images back to uint8 for saving
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, name), img)


def preprocess_frame(frame, step_idx=0):
    # --- original ---
    #save_debug_image(frame, f"{step_idx}_0_original.png")

    # --- crop ---
    h, w, _ = frame.shape
    #frame = frame[:h - h // 5, w // 7:, :]
    frame = frame[:h - h // 5, w // 7: - w // 7, :]
    
    #save_debug_image(frame, f"{step_idx}_1_cropped.png")

    # --- keep only bird + pipe borders ---
    frame, border_mask = keep_bird_and_pipe_borders(frame)
    #save_debug_image(frame, f"{step_idx}_2_bird_pipe.png")

    # --- save border mask for inspection ---
    #save_debug_image(border_mask.astype(np.uint8) * 255,
                     #f"{step_idx}_2b_border_mask.png")

    # --- grayscale (borders = white) ---
    frame = to_grayscale_with_white_borders(frame, border_mask)
    #save_debug_image(frame, f"{step_idx}_3_grayscale.png")

    # --- resize ---
    frame = cv2.resize(frame, (57, 57), interpolation=cv2.INTER_AREA)
    #save_debug_image(frame, f"{step_idx}_4_resized.png")

    # --- normalize ---
    frame = frame / 255.0
    #save_debug_image(frame, f"{step_idx}_5_normalized.png")

    return frame
