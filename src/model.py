import torch
import torch.nn as nn

import os
import torchvision.utils as vutils

import torch.nn.functional as F

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
    def __init__(self, frame_skip=4, n_actions=2):
        super().__init__()
        self.step = 0

        # ---------- CNN feature extractor ----------
        self.conv1 = nn.Conv2d(frame_skip, 32, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # ---------- Shared FC ----------
        self.fc = nn.Linear(64 * 6 * 6, 512)

        # ---------- Dueling heads ----------
        self.value = nn.Linear(512, 1)          # V(s)
        self.advantage = nn.Linear(512, n_actions)  # A(s,a)

    def forward(self, x):
        self.step += 1

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        value = self.value(x)                    # [B, 1]
        advantage = self.advantage(x)            # [B, A]

        # ---- Dueling aggregation ----
        q = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q
