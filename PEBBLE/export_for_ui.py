"""
Export trained PEBBLE trajectories as videos for UI
"""
import sys
sys.path.append('../preference_ui')

from trajectory_video_generator import TrajectoryVideoGenerator
import torch
from reward_model import RewardModel

def export_comparisons_for_ui(num_comparisons=50):
    """Export trajectory pairs from trained PEBBLE model"""
    
    # Load trained reward model
    reward_model = RewardModel(
        ds=17,  # Walker2d observation dim
        da=6,   # Walker2d action dim
        ensemble_size=3,
        size_segment=50
    )
    
    # Load checkpoint
    checkpoint_dir = 'logs/pebble'
    step = 1000  # Or your final training step
    reward_model.load(checkpoint_dir, step)
    
    print(f"✓ Loaded reward model from step {step}")
    
    # Get trajectory segments from reward model
    sa_t_1, sa_t_2, r_t_1, r_t_2 = reward_model.get_queries(
        mb_size=num_comparisons
    )
    
    trajectory_pairs = []
    
    for i in range(num_comparisons):
        segment_a = sa_t_1[i]
        segment_b = sa_t_2[i]
        
        states_a = segment_a[:, :reward_model.ds]
        actions_a = segment_a[:, reward_model.ds:]
        
        states_b = segment_b[:, :reward_model.ds]
        actions_b = segment_b[:, reward_model.ds:]
        
        trajectory_pairs.append({
            'states_a': states_a,
            'actions_a': actions_a,
            'states_b': states_b,
            'actions_b': actions_b
        })
    
    # Generate videos
    generator = TrajectoryVideoGenerator(env_name='Walker2d-v4')
    generator.generate_comparison_videos(
        trajectory_pairs,
        output_dir='../preference_ui/static/videos'
    )
    generator.close()
    
    print(f"✓ Exported {num_comparisons} comparisons")
    print("✓ Videos saved to: ../preference_ui/static/videos")

if __name__ == '__main__':
    export_comparisons_for_ui(num_comparisons=50)
