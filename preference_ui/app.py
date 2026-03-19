"""
Preference Collection UI for PEBBLE - Walker2D Environment
Comprehensive timing tracking and preference feedback collection

Authors: Yang Guo & Marzieh Ghayour
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import json
import csv
import os
from datetime import datetime
from pathlib import Path
import threading
import queue

app = Flask(__name__)

# Configuration
CONFIG = {
    'data_dir': 'trajectory_data',
    'output_dir': 'preference_data',
    'video_dir': 'static/videos',
    'participant_id': 'P01',  # Set this per participant
    'session_id': 1,
    'condition': 'baseline',  # baseline, time_aware_opaque, time_aware_transparent, explicit_confidence, revision_enabled
}

# Global state
current_comparison = {
    'comparison_id': None,
    'trajectory_a': None,
    'trajectory_b': None,
    'traj_a_reward': None,
    'traj_b_reward': None,
    'traj_a_video': None,
    'traj_b_video': None,
    'timestamps': {},
    'comparison_number': 0
}

comparison_queue = queue.Queue()
results_log = []

# Ensure directories exist
for dir_path in [CONFIG['data_dir'], CONFIG['output_dir'], CONFIG['video_dir']]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


class TimingTracker:
    """Track all timing events for a single comparison"""
    
    def __init__(self, comparison_id):
        self.comparison_id = comparison_id
        self.events = {}
        
    def mark(self, event_name):
        """Mark a timing event with current timestamp"""
        self.events[event_name] = datetime.now().isoformat()
        
    def duration(self, start_event, end_event):
        """Calculate duration between two events in seconds"""
        if start_event not in self.events or end_event not in self.events:
            return None
        
        start = datetime.fromisoformat(self.events[start_event])
        end = datetime.fromisoformat(self.events[end_event])
        return (end - start).total_seconds()
    
    def get_all_durations(self):
        """Calculate all relevant durations"""
        durations = {}
        
        # Total feedback time
        if 'trajectories_shown' in self.events and 'confidence_submitted' in self.events:
            durations['total_feedback_time'] = self.duration('trajectories_shown', 'confidence_submitted')
        
        # Initial viewing time (before any interaction)
        if 'trajectories_shown' in self.events and 'first_interaction' in self.events:
            durations['initial_viewing_time'] = self.duration('trajectories_shown', 'first_interaction')
        
        # Deliberation time (from prompt shown to preference selected)
        if 'prompt_shown' in self.events and 'preference_selected' in self.events:
            durations['deliberation_time'] = self.duration('prompt_shown', 'preference_selected')
        
        # Time to confidence report (from preference to confidence)
        if 'preference_selected' in self.events and 'confidence_submitted' in self.events:
            durations['confidence_reporting_time'] = self.duration('preference_selected', 'confidence_submitted')
        
        # Total replay time
        replay_count = sum(1 for key in self.events if 'replay_' in key)
        durations['replay_count'] = replay_count
        
        # Calculate time between first replay start and last replay end
        replay_starts = [v for k, v in self.events.items() if 'replay_start' in k]
        replay_ends = [v for k, v in self.events.items() if 'replay_end' in k]
        
        if replay_starts and replay_ends:
            first_replay = min(replay_starts)
            last_replay = max(replay_ends)
            start_dt = datetime.fromisoformat(first_replay)
            end_dt = datetime.fromisoformat(last_replay)
            durations['total_replay_time'] = (end_dt - start_dt).total_seconds()
        else:
            durations['total_replay_time'] = 0
            
        return durations
    
    def to_dict(self):
        """Export all timing data as dictionary"""
        return {
            'comparison_id': self.comparison_id,
            'events': self.events,
            'durations': self.get_all_durations()
        }


class ComparisonLogger:
    """Log all comparison data to CSV"""
    
    def __init__(self, output_dir, participant_id, session_id):
        self.output_dir = output_dir
        self.participant_id = participant_id
        self.session_id = session_id
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{output_dir}/{participant_id}_session{session_id}_{timestamp}_comparisons.csv"
        
        # CSV headers
        self.headers = [
            # Identifiers
            'comparison_id', 'participant_id', 'session_id', 'comparison_number',
            'timestamp', 'condition',
            
            # Trajectory info
            'traj_a_id', 'traj_b_id', 'traj_a_reward', 'traj_b_reward',
            'reward_difference', 'correct_choice', 'difficulty_category',
            
            # Preference
            'choice_made', 'choice_accuracy',
            
            # Confidence (if applicable)
            'confidence_score', 'confidence_method',
            
            # Timing - Total
            'total_feedback_time',
            
            # Timing - Phases
            'initial_viewing_time', 'deliberation_time', 
            'confidence_reporting_time', 'total_replay_time',
            
            # Replay behavior
            'replay_count', 'replay_trajectory_a', 'replay_trajectory_b',
            
            # Raw timestamps (for verification)
            'trajectories_shown_time', 'first_interaction_time',
            'prompt_shown_time', 'preference_selected_time',
            'confidence_submitted_time'
        ]
        
        # Create CSV with headers
        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
    
    def log_comparison(self, data):
        """Log a single comparison to CSV"""
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            
            # Fill in any missing fields with None
            row = {key: data.get(key, None) for key in self.headers}
            writer.writerow(row)
        
        print(f"✓ Logged comparison {data['comparison_id']}")


# Initialize logger
logger = ComparisonLogger(
    CONFIG['output_dir'],
    CONFIG['participant_id'],
    CONFIG['session_id']
)


@app.route('/')
def index():
    """Main comparison interface"""
    return render_template('index.html', 
                         condition=CONFIG['condition'],
                         participant_id=CONFIG['participant_id'])


@app.route('/api/get_comparison', methods=['GET'])
def get_comparison():
    """
    Fetch the next trajectory comparison
    
    Returns:
        JSON with trajectory data and timing initialization
    """
    global current_comparison
    
    # Increment comparison number
    current_comparison['comparison_number'] += 1
    comp_num = current_comparison['comparison_number']
    
    # Generate comparison ID
    comparison_id = f"{CONFIG['participant_id']}_S{CONFIG['session_id']}_C{comp_num:03d}"
    current_comparison['comparison_id'] = comparison_id
    
    # Initialize timing tracker
    current_comparison['timing'] = TimingTracker(comparison_id)
    current_comparison['timing'].mark('comparison_created')
    
    # Load trajectories (from queue or generate dummy data)
    # In production, this would load from saved trajectory files
    
    # DUMMY DATA FOR DEMO - Replace with actual trajectory loading
    traj_a_data = {
        'id': f'traj_a_{comp_num}',
        'video_path': f'/static/videos/walker_traj_a_{comp_num}.mp4',
        'reward': np.random.uniform(200, 400),
        'length': 50,  # timesteps
    }
    
    traj_b_data = {
        'id': f'traj_b_{comp_num}',
        'video_path': f'/static/videos/walker_traj_b_{comp_num}.mp4',
        'reward': np.random.uniform(200, 400),
        'length': 50,  # timesteps
    }
    
    current_comparison['trajectory_a'] = traj_a_data
    current_comparison['trajectory_b'] = traj_b_data
    current_comparison['traj_a_reward'] = traj_a_data['reward']
    current_comparison['traj_b_reward'] = traj_b_data['reward']
    
    # Calculate ground truth
    correct = 'A' if traj_a_data['reward'] > traj_b_data['reward'] else 'B'
    reward_diff = abs(traj_a_data['reward'] - traj_b_data['reward'])
    
    # Categorize difficulty
    if reward_diff > 50:
        difficulty = 'Easy'
    elif reward_diff > 20:
        difficulty = 'Medium'
    else:
        difficulty = 'Hard'
    
    current_comparison['correct_choice'] = correct
    current_comparison['difficulty'] = difficulty
    current_comparison['reward_difference'] = reward_diff
    
    # Initialize replay counters
    current_comparison['replay_a_count'] = 0
    current_comparison['replay_b_count'] = 0
    
    return jsonify({
        'comparison_id': comparison_id,
        'comparison_number': comp_num,
        'trajectory_a': traj_a_data,
        'trajectory_b': traj_b_data,
        'condition': CONFIG['condition'],
        'show_confidence': CONFIG['condition'] in ['explicit_confidence', 'baseline'],
        'show_timer': CONFIG['condition'] == 'time_aware_transparent'
    })


@app.route('/api/mark_timing', methods=['POST'])
def mark_timing():
    """
    Mark a timing event
    
    Expected JSON:
    {
        "event": "trajectories_shown" | "first_interaction" | "prompt_shown" | 
                 "preference_selected" | "confidence_submitted" | 
                 "replay_a_start_N" | "replay_a_end_N" | "replay_b_start_N" | "replay_b_end_N"
    }
    """
    data = request.json
    event_name = data.get('event')
    
    if current_comparison['timing']:
        current_comparison['timing'].mark(event_name)
        
        # Track replay counts
        if 'replay_a' in event_name:
            current_comparison['replay_a_count'] += 1
        elif 'replay_b' in event_name:
            current_comparison['replay_b_count'] += 1
        
        return jsonify({'status': 'success', 'event': event_name})
    
    return jsonify({'status': 'error', 'message': 'No active comparison'}), 400


@app.route('/api/submit_preference', methods=['POST'])
def submit_preference():
    """
    Submit preference and confidence score
    
    Expected JSON:
    {
        "choice": "A" | "B",
        "confidence": 1-5 (if applicable),
        "confidence_method": "explicit" | "implicit" | null
    }
    """
    data = request.json
    choice = data.get('choice')
    confidence = data.get('confidence', None)
    confidence_method = data.get('confidence_method', None)
    
    if not current_comparison['comparison_id']:
        return jsonify({'status': 'error', 'message': 'No active comparison'}), 400
    
    # Calculate accuracy
    accuracy = (choice == current_comparison['correct_choice'])
    
    # Get all timing data
    timing_data = current_comparison['timing'].to_dict()
    durations = timing_data['durations']
    events = timing_data['events']
    
    # Prepare log entry
    log_entry = {
        # Identifiers
        'comparison_id': current_comparison['comparison_id'],
        'participant_id': CONFIG['participant_id'],
        'session_id': CONFIG['session_id'],
        'comparison_number': current_comparison['comparison_number'],
        'timestamp': datetime.now().isoformat(),
        'condition': CONFIG['condition'],
        
        # Trajectory info
        'traj_a_id': current_comparison['trajectory_a']['id'],
        'traj_b_id': current_comparison['trajectory_b']['id'],
        'traj_a_reward': current_comparison['traj_a_reward'],
        'traj_b_reward': current_comparison['traj_b_reward'],
        'reward_difference': current_comparison['reward_difference'],
        'correct_choice': current_comparison['correct_choice'],
        'difficulty_category': current_comparison['difficulty'],
        
        # Preference
        'choice_made': choice,
        'choice_accuracy': accuracy,
        
        # Confidence
        'confidence_score': confidence,
        'confidence_method': confidence_method,
        
        # Timing - Total
        'total_feedback_time': durations.get('total_feedback_time'),
        
        # Timing - Phases
        'initial_viewing_time': durations.get('initial_viewing_time'),
        'deliberation_time': durations.get('deliberation_time'),
        'confidence_reporting_time': durations.get('confidence_reporting_time'),
        'total_replay_time': durations.get('total_replay_time'),
        
        # Replay behavior
        'replay_count': durations.get('replay_count', 0),
        'replay_trajectory_a': current_comparison['replay_a_count'],
        'replay_trajectory_b': current_comparison['replay_b_count'],
        
        # Raw timestamps
        'trajectories_shown_time': events.get('trajectories_shown'),
        'first_interaction_time': events.get('first_interaction'),
        'prompt_shown_time': events.get('prompt_shown'),
        'preference_selected_time': events.get('preference_selected'),
        'confidence_submitted_time': events.get('confidence_submitted'),
    }
    
    # Log to CSV
    logger.log_comparison(log_entry)
    
    # Store in memory for session summary
    results_log.append(log_entry)
    
    return jsonify({
        'status': 'success',
        'comparison_id': current_comparison['comparison_id'],
        'accuracy': accuracy,
        'total_comparisons': len(results_log),
        'timing_summary': durations
    })


@app.route('/api/session_summary', methods=['GET'])
def session_summary():
    """Get summary statistics for current session"""
    
    if not results_log:
        return jsonify({'status': 'error', 'message': 'No comparisons logged'}), 400
    
    # Calculate summary statistics
    total_comparisons = len(results_log)
    correct_count = sum(1 for r in results_log if r['choice_accuracy'])
    accuracy_rate = correct_count / total_comparisons if total_comparisons > 0 else 0
    
    # Timing statistics
    deliberation_times = [r['deliberation_time'] for r in results_log if r['deliberation_time']]
    total_feedback_times = [r['total_feedback_time'] for r in results_log if r['total_feedback_time']]
    replay_counts = [r['replay_count'] for r in results_log if r['replay_count'] is not None]
    
    summary = {
        'total_comparisons': total_comparisons,
        'correct_choices': correct_count,
        'accuracy_rate': f"{accuracy_rate:.1%}",
        
        'avg_deliberation_time': f"{np.mean(deliberation_times):.2f}s" if deliberation_times else "N/A",
        'median_deliberation_time': f"{np.median(deliberation_times):.2f}s" if deliberation_times else "N/A",
        'std_deliberation_time': f"{np.std(deliberation_times):.2f}s" if deliberation_times else "N/A",
        
        'avg_total_feedback_time': f"{np.mean(total_feedback_times):.2f}s" if total_feedback_times else "N/A",
        
        'total_replays': sum(replay_counts) if replay_counts else 0,
        'avg_replays_per_comparison': f"{np.mean(replay_counts):.1f}" if replay_counts else "0",
        
        'comparisons_with_replays': sum(1 for r in replay_counts if r > 0) if replay_counts else 0,
    }
    
    return jsonify(summary)


@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    """Serve video files"""
    return send_from_directory(CONFIG['video_dir'], filename)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PEBBLE Preference Collection UI")
    print("="*60)
    print(f"Participant ID: {CONFIG['participant_id']}")
    print(f"Session ID: {CONFIG['session_id']}")
    print(f"Condition: {CONFIG['condition']}")
    print(f"Output: {logger.filename}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
