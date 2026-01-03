import os
import hashlib
import torch
import random
from gymnasium.wrappers import RecordVideo
from collections import deque
import gymnasium as gym
import numpy as np
from model import DQN_CNN
from preprocessing.preprocessing import preprocess_frame
from PIL import Image
from static_variables import PHOTOS_DIR

frame_id = 0

def save_model(
    policy_net,
    target_net,
    optimizer,
    dir,
    global_step=None,
    best_reward=None,
):
    h = hashlib.md5(random.randbytes(16)).hexdigest()
    path = f"{dir}/{h}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "policy_net_state_dict": policy_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "best_reward": best_reward,
    }

    torch.save(checkpoint, path)
    print(f"✅ Model saved to {path}")

    return path


def load_model(
    policy_net,
    target_net,
    optimizer,
    filepath,
    device,
):
    checkpoint = torch.load(filepath, map_location=device)

    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    target_net.load_state_dict(checkpoint["target_net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    policy_net.to(device)
    target_net.to(device)

    global_step = checkpoint.get("global_step", 0)
    best_reward = checkpoint.get("best_reward", None)

    print(f"✅ Model loaded from {filepath}")

    return global_step, best_reward

def transition(action, env, frame_skip):
    global frame_id

    frame_stack = []
    total_reward = 0
    done = False

    _, reward, terminated, truncated, _ = env.step(action)
    frame = preprocess_frame(env.render())
    frame_stack.append(frame)

    total_reward += reward
    done = terminated or truncated

    for _ in range(frame_skip - 1):
        if done:
            frame_stack.append(frame)  # pad with last frame
            continue

        _, reward, terminated, truncated, _ = env.step(0)
        frame = preprocess_frame(env.render())
        frame_stack.append(frame)

        total_reward += reward
        done = terminated or truncated

        # Image.fromarray(frame).save(
        #     f"{PHOTOS_DIR}/frame_{'died_' if done else ''}{frame_id:05d}.png"
        # )
        frame_id += 1

    return np.stack(frame_stack), total_reward, done


def record_trained_agent_video(
    model_path="checkpoints/flappy_dqn.pt",
    video_dir="videos",
    env_name="FlappyBird-v0",
    frame_skip=4,
    device="cpu",
):
    os.makedirs(video_dir, exist_ok=True)

    # ---------- environment with video recording ----------
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,  # record first episode
        name_prefix="flappy_dqn",
    )

    # ---------- load model ----------
    policy_net = DQN_CNN(frame_skip).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    policy_net.eval()

    # ---------- reset ----------
    env.reset()
    frame_stack = deque(maxlen=frame_skip)

    done = False
    total_reward = 0

    # ---------- play episode ----------
    current_state, _, _ = transition(0, env, frame_skip)

    while not done:
        with torch.no_grad():
            state_t = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(device)
            action = policy_net(state_t).argmax(dim=1).item()

        next_state, reward_sum, done = transition(action, env, frame_skip)
        current_state = next_state

        total_reward += reward_sum

    env.close()
    print(f"🎥 Video saved in '{video_dir}/' | Reward: {total_reward:.2f}")


