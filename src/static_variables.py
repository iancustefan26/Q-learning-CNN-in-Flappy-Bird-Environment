import os

PHOTOS_DIR = "../photos"
VIDEOS_DIR = "../videos"
CHECKPOINTS_DIR = "../checkpoints"

def make_dirs():
    """Create necessary directories if they don't exist."""
    if not os.path.exists(PHOTOS_DIR):
        os.makedirs(PHOTOS_DIR)

    if not os.path.exists(VIDEOS_DIR):
        os.makedirs(VIDEOS_DIR)

    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)    