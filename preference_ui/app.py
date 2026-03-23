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
    'use_query_bank': True,
    'query_job_id': '',  # e.g., 4422468 (uses ../human_queries/{job_id}/...)
    'query_root_dir': '../human_queries/{job_id}',
    'query_manifest_csv': '../human_queries/{job_id}/query_manifest.csv',
    # You can use placeholders: {participant_id}, {session_id}, {job_id}
    # Example: ../human_queries/{job_id}/human_labels_{participant_id}_s{session_id}.csv
    'query_labels_csv_template': '../human_queries/{job_id}/human_labels_{participant_id}_s{session_id}.csv',
    'query_labels_csv': '../human_queries/human_labels_{participant_id}_s{session_id}.csv',
    'skip_already_labeled': True,
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
query_bank = []
query_bank_index = 0
logger = None

# Ensure directories exist
for dir_path in [CONFIG['data_dir'], CONFIG['output_dir'], CONFIG['video_dir']]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def _resolve_config_paths():
    """Resolve template placeholders in configurable paths."""
    def _fmt(v):
        return str(v).format(
            participant_id=CONFIG.get('participant_id', 'PXX'),
            session_id=CONFIG.get('session_id', 1),
            job_id=CONFIG.get('query_job_id', ''),
        )

    if 'query_root_dir' in CONFIG:
        CONFIG['query_root_dir'] = _fmt(CONFIG['query_root_dir'])
    if 'query_manifest_csv' in CONFIG:
        CONFIG['query_manifest_csv'] = _fmt(CONFIG['query_manifest_csv'])

    template = CONFIG.get('query_labels_csv_template', CONFIG.get('query_labels_csv', ''))
    if template:
        CONFIG['query_labels_csv'] = _fmt(template)

    Path(CONFIG['query_root_dir']).mkdir(parents=True, exist_ok=True)


def _normalize_rel_path(p):
    return str(p).replace('\\', '/').lstrip('./')


def _init_human_labels_csv(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'query_id',
                'label',
                'choice',
                'confidence',
                'confidence_method',
                'participant_id',
                'session_id',
                'comparison_number',
                'timestamp',
                'condition',
            ])


def _load_existing_labeled_query_ids(path):
    if not os.path.exists(path):
        return set()
    labeled = set()
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return labeled
        if 'query_id' not in reader.fieldnames:
            return labeled
        for row in reader:
            qid = str(row.get('query_id', '')).strip()
            label = str(row.get('label', '')).strip()
            if qid and label != '':
                labeled.add(qid)
    return labeled


def _load_query_bank_from_manifest():
    manifest_path = CONFIG['query_manifest_csv']
    if not os.path.exists(manifest_path):
        print(f"⚠️ Query manifest not found: {manifest_path}")
        return []

    labeled_ids = set()
    if CONFIG.get('skip_already_labeled', True):
        labeled_ids = _load_existing_labeled_query_ids(CONFIG['query_labels_csv'])

    rows = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        required = {'query_id', 'video_a_path', 'video_b_path'}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Manifest must contain columns: {sorted(required)}. Got: {reader.fieldnames}"
            )

        for row in reader:
            qid = str(row.get('query_id', '')).strip()
            if not qid:
                continue
            if qid in labeled_ids:
                continue

            video_a_rel = _normalize_rel_path(row.get('video_a_path', ''))
            video_b_rel = _normalize_rel_path(row.get('video_b_path', ''))
            if not video_a_rel or not video_b_rel:
                continue

            row['video_a_path'] = video_a_rel
            row['video_b_path'] = video_b_rel
            rows.append(row)

    print(f"Loaded query bank: {len(rows)} unlabeled queries")
    return rows


def _append_human_label(query_id, choice, confidence, confidence_method, comparison_number):
    label_value = 0 if choice == 'A' else 1 if choice == 'B' else ''
    with open(CONFIG['query_labels_csv'], 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            query_id,
            label_value,
            choice,
            confidence if confidence is not None else '',
            confidence_method if confidence_method is not None else '',
            CONFIG['participant_id'],
            CONFIG['session_id'],
            comparison_number,
            datetime.now().isoformat(),
            CONFIG['condition'],
        ])


def _initialize_runtime_state(reset_results=True):
    global logger, query_bank, query_bank_index

    _resolve_config_paths()

    logger = ComparisonLogger(
        CONFIG['output_dir'],
        CONFIG['participant_id'],
        CONFIG['session_id']
    )

    query_bank_index = 0
    if CONFIG.get('use_query_bank', False):
        _init_human_labels_csv(CONFIG['query_labels_csv'])
        query_bank = _load_query_bank_from_manifest()
    else:
        query_bank = []

    current_comparison['comparison_id'] = None
    current_comparison['trajectory_a'] = None
    current_comparison['trajectory_b'] = None
    current_comparison['traj_a_reward'] = None
    current_comparison['traj_b_reward'] = None
    current_comparison['traj_a_video'] = None
    current_comparison['traj_b_video'] = None
    current_comparison['timestamps'] = {}
    current_comparison['comparison_number'] = 0

    if reset_results:
        results_log.clear()


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


_initialize_runtime_state(reset_results=True)


@app.route('/')
def index():
    """Main comparison interface"""
    return render_template('index.html', 
                         condition=CONFIG['condition'],
                         participant_id=CONFIG['participant_id'])


@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Set participant/session values and reset runtime state for a new UI session."""
    data = request.json or {}

    participant_id = str(data.get('participant_id', '')).strip()
    if not participant_id:
        return jsonify({'status': 'error', 'message': 'participant_id is required'}), 400

    raw_session = str(data.get('session_id', '')).strip()
    if not raw_session:
        return jsonify({'status': 'error', 'message': 'session_id is required'}), 400

    try:
        session_id = int(raw_session)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'session_id must be an integer'}), 400

    CONFIG['participant_id'] = participant_id
    CONFIG['session_id'] = session_id
    _initialize_runtime_state(reset_results=True)

    return jsonify({
        'status': 'success',
        'participant_id': CONFIG['participant_id'],
        'session_id': CONFIG['session_id'],
        'query_labels_csv': CONFIG.get('query_labels_csv'),
        'remaining_queries': len(query_bank),
    })


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
    
    if CONFIG.get('use_query_bank', False):
        global query_bank_index
        if query_bank_index >= len(query_bank):
            return jsonify({'status': 'complete', 'message': 'No more unlabeled queries'}), 410

        q = query_bank[query_bank_index]
        query_bank_index += 1
        current_comparison['query_id'] = q['query_id']

        traj_a_data = {
            'id': f"{q['query_id']}_A",
            'video_path': f"/query_media/{q['video_a_path']}",
            'reward': None,
            'length': int(q.get('segment_length', 50) or 50),
        }

        traj_b_data = {
            'id': f"{q['query_id']}_B",
            'video_path': f"/query_media/{q['video_b_path']}",
            'reward': None,
            'length': int(q.get('segment_length', 50) or 50),
        }

        current_comparison['trajectory_a'] = traj_a_data
        current_comparison['trajectory_b'] = traj_b_data
        current_comparison['traj_a_reward'] = None
        current_comparison['traj_b_reward'] = None
        current_comparison['correct_choice'] = None
        current_comparison['difficulty'] = 'Unknown'
        current_comparison['reward_difference'] = None
    else:
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
    
    # Calculate accuracy (if available)
    accuracy = None
    if current_comparison.get('correct_choice') in ['A', 'B']:
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

    # Write offline human labels file (query_id,label,confidence,...) for PEBBLE training
    if CONFIG.get('use_query_bank', False):
        query_id = current_comparison.get('query_id')
        if query_id:
            _append_human_label(
                query_id=query_id,
                choice=choice,
                confidence=confidence,
                confidence_method=confidence_method,
                comparison_number=current_comparison['comparison_number'],
            )
    
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


@app.route('/query_media/<path:filename>')
def serve_query_media(filename):
    """Serve query-bank videos from query root directory"""
    return send_from_directory(CONFIG['query_root_dir'], filename)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PEBBLE Preference Collection UI")
    print("="*60)
    print(f"Participant ID: {CONFIG['participant_id']}")
    print(f"Session ID: {CONFIG['session_id']}")
    print(f"Condition: {CONFIG['condition']}")
    print(f"Output: {logger.filename}")
    if CONFIG.get('use_query_bank', False):
        print(f"Query manifest: {CONFIG['query_manifest_csv']}")
        print(f"Human labels output: {CONFIG['query_labels_csv']}")
        print(f"Unlabeled queries available: {len(query_bank)}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
