"""
Trajectory Video Generator for Walker2D Environment
Converts trajectory state-action sequences to MP4 videos for UI

Authors: Yang Guo & Marzieh Ghayour
"""

import gymnasium as gym
import numpy as np
import cv2
import os
from pathlib import Path
import json
from tqdm import tqdm


class TrajectoryVideoGenerator:
    """Generate videos from Walker2D trajectories"""
    
    def __init__(self, env_name='Walker2d-v4', video_fps=30, video_size=(640, 480)):
        """
        Args:
            env_name: Gymnasium environment name
            video_fps: Frames per second for output video
            video_size: (width, height) of output video
        """
        self.env_name = env_name
        self.video_fps = video_fps
        self.video_size = video_size
        
        # Create environment with rendering
        self.env = gym.make(env_name, render_mode='rgb_array')
        
    def render_trajectory(self, states, actions):
        """
        Render a trajectory to frames
        
        Args:
            states: np.array of shape (timesteps, state_dim)
            actions: np.array of shape (timesteps, action_dim)
            
        Returns:
            frames: List of RGB frames
            total_reward: Cumulative reward
        """
        frames = []
        total_reward = 0
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Execute trajectory
        for t in range(len(actions)):
            # Get frame before action
            frame = self.env.render()
            frames.append(frame)
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(actions[t])
            total_reward += reward
            
            if terminated or truncated:
                pass  # Keep rendering
        
        return frames, total_reward
    
    def save_video(self, frames, output_path, fourcc='mp4v'):
        """
        Save frames as MP4 video
        
        Args:
            frames: List of RGB frames (numpy arrays)
            output_path: Path to save video
            fourcc: Video codec (mp4v, avc1, etc.)
        """
        if not frames:
            print(f"⚠️  No frames to save for {output_path}")
            return
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        out = cv2.VideoWriter(
            output_path,
            fourcc_code,
            self.video_fps,
            self.video_size
        )
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            if frame_bgr.shape[:2][::-1] != self.video_size:
                frame_bgr = cv2.resize(frame_bgr, self.video_size)
            
            out.write(frame_bgr)
        
        out.release()
        print(f"✓ Saved video: {output_path}")
    
    def generate_comparison_videos(self, 
                                   trajectory_pairs,
                                   output_dir='static/videos',
                                   start_idx=1):
        """
        Generate video pairs for comparisons
        
        Args:
            trajectory_pairs: List of dicts with keys:
                - 'states_a': np.array of states for trajectory A
                - 'actions_a': np.array of actions for trajectory A
                - 'states_b': np.array of states for trajectory B
                - 'actions_b': np.array of actions for trajectory B
            output_dir: Directory to save videos
            start_idx: Starting index for numbering
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        metadata_list = []
        
        for i, pair in enumerate(tqdm(trajectory_pairs, desc="Generating videos")):
            comp_idx = start_idx + i
            
            # Generate trajectory A video
            frames_a, reward_a = self.render_trajectory(
                pair['states_a'], 
                pair['actions_a']
            )
            video_path_a = f"{output_dir}/walker_traj_a_{comp_idx}.mp4"
            self.save_video(frames_a, video_path_a)
            
            # Reset environment between trajectories
            self.env.reset()
            
            # Generate trajectory B video
            frames_b, reward_b = self.render_trajectory(
                pair['states_b'],
                pair['actions_b']
            )
            video_path_b = f"{output_dir}/walker_traj_b_{comp_idx}.mp4"
            self.save_video(frames_b, video_path_b)
            
            # Calculate ground truth
            correct = 'A' if reward_a > reward_b else 'B'
            reward_diff = abs(reward_a - reward_b)
            
            # Categorize difficulty
            if reward_diff > 50:
                difficulty = 'Easy'
            elif reward_diff > 20:
                difficulty = 'Medium'
            else:
                difficulty = 'Hard'
            
            # Store metadata
            metadata = {
                'comparison_id': comp_idx,
                'video_a_path': video_path_a,
                'video_b_path': video_path_b,
                'reward_a': float(reward_a),
                'reward_b': float(reward_b),
                'reward_difference': float(reward_diff),
                'correct_choice': correct,
                'difficulty': difficulty,
                'frames_a': len(frames_a),
                'frames_b': len(frames_b)
            }
            metadata_list.append(metadata)
            
            # Save individual metadata
            metadata_path = f"{output_dir}/comparison_{comp_idx}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save combined metadata
        combined_path = f"{output_dir}/all_comparisons_metadata.json"
        with open(combined_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        print(f"\n✓ Generated {len(trajectory_pairs)} comparison pairs")
        print(f"✓ Metadata saved to: {combined_path}")
    
    def close(self):
        """Clean up environment"""
        self.env.close()


def load_trajectories_from_pebble(reward_model, num_comparisons=50, segment_length=50):
    """
    Load trajectory pairs from PEBBLE reward model
    
    Args:
        reward_model: PEBBLE RewardModel instance
        num_comparisons: Number of comparison pairs to generate
        segment_length: Length of each trajectory segment
        
    Returns:
        trajectory_pairs: List of trajectory pairs
    """
    # Get trajectory segments from reward model
    sa_t_1, sa_t_2, r_t_1, r_t_2 = reward_model.get_queries(mb_size=num_comparisons)
    
    trajectory_pairs = []
    
    for i in range(num_comparisons):
        # Extract state and action from concatenated sa
        # Assuming states are first ds dimensions, actions are next da dimensions
        segment_a = sa_t_1[i]  # shape: (segment_length, ds + da)
        segment_b = sa_t_2[i]
        
        states_a = segment_a[:, :reward_model.ds]
        actions_a = segment_a[:, reward_model.ds:]
        
        states_b = segment_b[:, :reward_model.ds]
        actions_b = segment_b[:, reward_model.ds:]
        
        pair = {
            'states_a': states_a,
            'actions_a': actions_a,
            'states_b': states_b,
            'actions_b': actions_b
        }
        
        trajectory_pairs.append(pair)
    
    return trajectory_pairs


def generate_dummy_trajectories(num_pairs=10, segment_length=50):
    """
    Generate dummy trajectories for testing (random actions)
    
    Args:
        num_pairs: Number of comparison pairs
        segment_length: Length of each trajectory
        
    Returns:
        trajectory_pairs: List of trajectory pairs
    """
    env = gym.make('Walker2d-v4')
    
    trajectory_pairs = []
    
    for _ in range(num_pairs):
        # Trajectory A
        states_a = []
        actions_a = []
        
        obs, _ = env.reset()
        for _ in range(segment_length):
            action = env.action_space.sample()  # Random action
            states_a.append(obs)
            actions_a.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()  # Reset instead of breaking
                # 
        
        # Trajectory B
        states_b = []
        actions_b = []
        
        obs, _ = env.reset()
        for _ in range(segment_length):
            action = env.action_space.sample()
            states_b.append(obs)
            actions_b.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()  # Reset instead of breaking
                # 
        
        pair = {
            'states_a': np.array(states_a),
            'actions_a': np.array(actions_a),
            'states_b': np.array(states_b),
            'actions_b': np.array(actions_b)
        }
        
        trajectory_pairs.append(pair)
    
    env.close()
    return trajectory_pairs


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Walker2D Trajectory Video Generator")
    print("="*60 + "\n")
    
    # Configuration
    NUM_COMPARISONS = 10
    SEGMENT_LENGTH = 50
    OUTPUT_DIR = 'static/videos'
    
    print(f"Generating {NUM_COMPARISONS} comparison pairs...")
    print(f"Segment length: {SEGMENT_LENGTH} timesteps")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Generate dummy trajectories for demo
    # In production, replace with: load_trajectories_from_pebble(reward_model, ...)
    trajectories = generate_dummy_trajectories(
        num_pairs=NUM_COMPARISONS,
        segment_length=SEGMENT_LENGTH
    )
    
    # Create video generator
    generator = TrajectoryVideoGenerator(
        env_name='Walker2d-v4',
        video_fps=30,
        video_size=(800, 600)
    )
    
    # Generate videos
    try:
        generator.generate_comparison_videos(
            trajectories,
            output_dir=OUTPUT_DIR,
            start_idx=1
        )
    finally:
        generator.close()
    
    print("\n" + "="*60)
    print("✓ Video generation complete!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("1. Check videos in:", OUTPUT_DIR)
    print("2. Update app.py CONFIG with participant details")
    print("3. Run: python app.py")
    print("4. Open: http://localhost:5000")
    print()
