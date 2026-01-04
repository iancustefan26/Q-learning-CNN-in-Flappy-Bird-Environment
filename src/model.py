import torch
import torch.nn as nn

import os
import torchvision.utils as vutils

def save_input_frames(x, filename):
    """
    x: (B, frame_skip, H, W)
    Saves the stacked input frames as a grid
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    frames = x[0]                  # (frame_skip, H, W)
    frames = frames.unsqueeze(1)   # (frame_skip, 1, H, W)

    grid = vutils.make_grid(
        frames,
        nrow=frames.shape[0],
        normalize=True,
        scale_each=True
    )

    vutils.save_image(grid, filename)

def save_feature_maps(x, filename, max_channels=32):
    """
    x: (B, C, H, W)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fmap = x[0][:max_channels]     # (C, H, W)
    fmap = fmap.unsqueeze(1)       # (C, 1, H, W)

    grid = vutils.make_grid(
        fmap,
        nrow=8,
        normalize=True,
        scale_each=True
    )

    vutils.save_image(grid, filename)

def vis_hook(name, step_getter):
    def hook(module, input, output):
        step = step_getter()
        save_feature_maps(
            output.detach().cpu(),
            f"vis/step_{step:06d}_{name}.png"
        )
    return hook

class DQN_CNN(nn.Module):
    def __init__(self, frame_skip=4):
        super().__init__()
        self.step = 0

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d( in_channels=frame_skip, 
                                out_channels=6,
                                kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(8 * 21 * 21, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        # hooks
       #self.conv1.register_forward_hook(vis_hook("conv1", lambda: self.step))
        #self.conv2.register_forward_hook(vis_hook("conv2", lambda: self.step))

        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,

            self.conv2,
            nn.ReLU(),
            self.pool,

            nn.Flatten(),

            self.fc1,
            nn.LayerNorm(128),
            nn.ReLU(),

            self.fc2,
            nn.LayerNorm(64),
            nn.ReLU(),

            self.fc3
        )

    def forward(self, x):
        self.step += 1
        return self.net(x)
