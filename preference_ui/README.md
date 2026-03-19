# PEBBLE Preference Collection UI for Walker2D

Comprehensive web-based UI for collecting human preferences on trajectory pairs with detailed timing tracking.

**Authors:** Yang Guo & Marzieh Ghayour  
**Project:** Human Correction in Interactive Reinforcement Learning: Incorporating Uncertainty-Aware Feedback

---

## Features

### ✅ Comprehensive Timing Tracking

1. **Total Feedback Time**: Full duration from trajectories shown to confidence submitted
2. **Initial Viewing Time**: Time before first user interaction
3. **Deliberation Time**: Time from prompt shown to preference selected
4. **Confidence Reporting Time**: Time from preference to confidence submission
5. **Replay Tracking**: Count and duration of trajectory replays

### ✅ User Interface

- **Side-by-side trajectory comparison**
- **Replay buttons** for each trajectory (disabled until initial viewing complete)
- **Clear preference selection** (Trajectory A vs B)
- **Confidence rating** (1-5 scale, conditionally shown)
- **Real-time timer** (for transparent condition only)
- **Responsive design** (works on desktop and tablet)

### ✅ Experimental Conditions Support

- **Baseline**: Standard preference collection
- **Time-Aware (Opaque)**: Time tracked but user not informed
- **Time-Aware (Transparent)**: User sees live timer
- **Explicit Confidence**: User provides confidence scores
- **Revision-Enabled**: Allows post-hoc corrections

### ✅ Data Logging

All data automatically logged to CSV with comprehensive metrics:
- Timing data (deliberation, replay, total)
- Preference choices and accuracy
- Confidence scores
- Replay behavior
- Ground truth comparisons

---

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install flask numpy pandas --break-system-packages
```

### Directory Structure

```
preference_ui/
├── app.py                      # Flask backend
├── templates/
│   └── index.html             # Main UI template
├── static/
│   ├── css/
│   │   └── style.css          # Comprehensive styling
│   ├── js/
│   │   └── app.js             # Frontend application
│   └── videos/                # Trajectory videos
│       ├── walker_traj_a_1.mp4
│       ├── walker_traj_b_1.mp4
│       └── ...
├── trajectory_data/            # Source trajectory files
├── preference_data/            # Output CSV logs
└── README.md
```

---

## Quick Start

### 1. Configure Participant

Edit `app.py` to set participant details:

```python
CONFIG = {
    'participant_id': 'P01',     # Change per participant
    'session_id': 1,             # Increment for each session
    'condition': 'baseline',     # See conditions below
}
```

**Available conditions:**
- `baseline` - Standard PEBBLE
- `time_aware_opaque` - Time tracked, user unaware
- `time_aware_transparent` - Time tracked, user informed
- `explicit_confidence` - User provides confidence scores
- `revision_enabled` - Post-hoc revision allowed

### 2. Generate Trajectory Videos

```bash
# You need to create videos from your Walker2D trajectories
# See trajectory_video_generator.py for automation

python trajectory_video_generator.py
```

### 3. Run the Server

```bash
cd preference_ui
python app.py
```

### 4. Open in Browser

Navigate to:
```
http://localhost:5000
```

---

## Usage

### For Participants

1. **Watch both trajectories** as they auto-play
2. **Replay if needed** using the replay buttons
3. **Choose which is better** (A or B)
4. **Rate your confidence** (if applicable, 1-5 scale)
5. **Proceed to next comparison**

### Keyboard Shortcuts

- `A` / `B` - Choose trajectory
- `1-5` - Confidence score
- `Space` / `Enter` - Next comparison
- `Ctrl+S` - Show session summary
- `Ctrl+D` - Toggle debug panel

---

## Data Output

### CSV Format

Logged to: `preference_data/P{id}_session{num}_{timestamp}_comparisons.csv`

**Key columns:**

```csv
comparison_id,participant_id,session_id,comparison_number,
traj_a_reward,traj_b_reward,correct_choice,choice_made,choice_accuracy,
confidence_score,
total_feedback_time,deliberation_time,initial_viewing_time,
confidence_reporting_time,total_replay_time,replay_count,
replay_trajectory_a,replay_trajectory_b,
trajectories_shown_time,preference_selected_time,...
```

### Timing Metrics

All times in **seconds**:

| Metric | Description |
|--------|-------------|
| `total_feedback_time` | Trajectories shown → Confidence submitted |
| `initial_viewing_time` | Trajectories shown → First interaction |
| `deliberation_time` | Prompt shown → Preference selected |
| `confidence_reporting_time` | Preference selected → Confidence submitted |
| `total_replay_time` | Sum of all replay durations |

### Example Log Entry

```csv
P01_S1_C001,P01,1,1,342.5,287.3,A,A,True,4,
32.7,3.5,2.1,1.8,0,0,0,0,
2026-03-18T14:32:17.483,2026-03-18T14:32:51.201,...
```

**Interpretation:**
- Comparison took 32.7s total
- User deliberated 3.5s before choosing
- Chose correctly (A)
- Confidence: 4/5
- No replays

---

## API Endpoints

### GET `/api/get_comparison`

Fetch next trajectory comparison.

**Response:**
```json
{
  "comparison_id": "P01_S1_C001",
  "comparison_number": 1,
  "trajectory_a": {
    "id": "traj_a_1",
    "video_path": "/static/videos/walker_traj_a_1.mp4",
    "reward": 342.5
  },
  "trajectory_b": { ... },
  "condition": "baseline",
  "show_confidence": true
}
```

### POST `/api/mark_timing`

Mark a timing event.

**Request:**
```json
{
  "event": "trajectories_shown" | "first_interaction" | 
           "prompt_shown" | "preference_selected" | 
           "confidence_submitted" | "replay_a_start_1" | ...
}
```

### POST `/api/submit_preference`

Submit preference and confidence.

**Request:**
```json
{
  "choice": "A" | "B",
  "confidence": 1-5 (optional),
  "confidence_method": "explicit" | null
}
```

**Response:**
```json
{
  "status": "success",
  "comparison_id": "P01_S1_C001",
  "accuracy": true,
  "total_comparisons": 1,
  "timing_summary": {
    "total_feedback_time": 32.7,
    "deliberation_time": 3.5,
    ...
  }
}
```

### GET `/api/session_summary`

Get session statistics.

**Response:**
```json
{
  "total_comparisons": 50,
  "accuracy_rate": "78.0%",
  "avg_deliberation_time": "4.23s",
  "total_replays": 12,
  ...
}
```

---

## Integration with PEBBLE Training

### 1. Generate Trajectory Pairs

From your PEBBLE `reward_model.py`:

```python
# In RewardModel class
def export_comparison_videos(self, save_dir, num_comparisons=50):
    """Export trajectory pairs as videos for UI"""
    
    sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=num_comparisons)
    
    for i in range(num_comparisons):
        # Trajectory A
        segment_a = sa_t_1[i]  # shape: (segment_length, obs_dim + action_dim)
        reward_a = r_t_1[i].sum()
        
        # Trajectory B
        segment_b = sa_t_2[i]
        reward_b = r_t_2[i].sum()
        
        # Render to video
        self.render_trajectory_to_video(
            segment_a, 
            f"{save_dir}/walker_traj_a_{i+1}.mp4"
        )
        self.render_trajectory_to_video(
            segment_b,
            f"{save_dir}/walker_traj_b_{i+1}.mp4"
        )
        
        # Save metadata
        metadata = {
            'comparison_id': i+1,
            'traj_a_reward': float(reward_a),
            'traj_b_reward': float(reward_b),
            'correct': 'A' if reward_a > reward_b else 'B'
        }
        
        with open(f"{save_dir}/comparison_{i+1}_metadata.json", 'w') as f:
            json.dump(metadata, f)
```

### 2. Load Human Preferences

After collecting preferences via UI:

```python
import pandas as pd

# Load logged preferences
df = pd.read_csv('preference_data/P01_session1_..._comparisons.csv')

# Extract preference labels
labels = []
for _, row in df.iterrows():
    label = 0 if row['choice_made'] == 'A' else 1
    labels.append(label)

# Optional: Weight by confidence or deliberation time
if 'confidence_score' in df.columns:
    weights = df['confidence_score'] / 5.0  # Normalize to [0, 1]
elif 'deliberation_time' in df.columns:
    # Inverse weighting: faster = more confident
    weights = 1.0 / (1.0 + df['deliberation_time'])
else:
    weights = np.ones(len(df))

# Put into PEBBLE reward model
reward_model.put_queries(sa_t_1, sa_t_2, labels, weights=weights)
```

---

## Customization

### Changing Video Duration

In `app.py`, adjust segment length:

```python
# Longer segments for more context
cfg.segment = 100  # timesteps (default: 50)
```

### Custom Confidence Scale

Modify `templates/index.html`:

```html
<!-- Change from 1-5 to 1-7 -->
<button class="confidence-btn" data-value="1">1</button>
...
<button class="confidence-btn" data-value="7">7</button>
```

### Adding Task Instructions

Edit condition message in `static/js/app.js`:

```javascript
case 'baseline':
    message = 'Your custom instructions here...';
    break;
```

---

## Troubleshooting

### Videos Not Playing

**Issue:** Browser blocks autoplay  
**Fix:** Click play button manually, or serve over HTTPS

### Timing Data Missing

**Issue:** Some timing events not logged  
**Fix:** Check browser console for errors; verify network connection

### CSV Not Updating

**Issue:** File permissions  
**Fix:** Check write permissions on `preference_data/` directory

---

## Citation

If you use this UI in your research, please cite:

```bibtex
@inproceedings{guo2026uncertainty,
  title={Human Correction in Interactive Reinforcement Learning: 
         Incorporating Uncertainty-Aware Feedback},
  author={Guo, Yang and Ghayour, Marzieh},
  booktitle={Conference TBD},
  year={2026}
}
```

---

## License

MIT License - see LICENSE file

---

## Contact

**Yang Guo** - University of Alberta  
**Marzieh Ghayour** - University of Alberta

For issues or questions, please contact the authors or open an issue on the repository.

---

## Acknowledgments

- Built on [PEBBLE](https://github.com/pokaxpoka/B_Pref) framework
- Uses [Walker2D-v4](https://gymnasium.farama.org/) from Gymnasium
- Inspired by preference learning research in interactive RL
