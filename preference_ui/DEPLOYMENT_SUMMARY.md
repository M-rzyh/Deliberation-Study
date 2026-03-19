# PEBBLE Preference Collection UI - Complete Package

## 🎯 What You Requested

A UI for rendering Walker2D trajectories and collecting preference feedback with comprehensive timing tracking:

1. ✅ **Trajectory Rendering** - Side-by-side video display
2. ✅ **Comprehensive Timing Tracking:**
   - Total feedback time
   - Initial viewing time
   - Deliberation time (substituting rewatch as primary metric)
   - Confidence reporting time
   - Replay/rewatch tracking
3. ✅ **Replay Buttons** - Per-trajectory replay with counter
4. ✅ **Confidence Scores** - 1-5 Likert scale (self-reported)

---

## 📦 What's Included

### Core Files

```
preference_ui/
├── app.py                          # Flask backend (16 KB)
├── templates/
│   └── index.html                  # Main UI (6 KB)
├── static/
│   ├── css/
│   │   └── style.css              # Comprehensive styling (15 KB)
│   └── js/
│       └── app.js                 # Frontend application (15 KB)
├── trajectory_video_generator.py   # Video generation utility (11 KB)
├── requirements.txt                # Python dependencies
├── setup.sh                        # Setup script (executable)
├── README.md                       # Complete documentation (10 KB)
└── TIMING_REFERENCE.md            # Timing system guide (9 KB)
```

### Directories (Auto-created)

```
static/videos/          # Trajectory videos (.mp4)
trajectory_data/        # Source trajectory files
preference_data/        # Output CSV logs
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd preference_ui
pip install -r requirements.txt --break-system-packages
```

Or use the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Generate Demo Videos (Optional)

```bash
python3 trajectory_video_generator.py
```

This creates 10 demo comparisons with random Walker2D policies.

### 3. Configure Participant

Edit `app.py`, lines 11-17:

```python
CONFIG = {
    'participant_id': 'P01',     # Change for each participant
    'session_id': 1,             # Increment per session
    'condition': 'baseline',     # See conditions below
}
```

**Available conditions:**
- `baseline` - Standard PEBBLE
- `time_aware_opaque` - Time tracked, user unaware
- `time_aware_transparent` - User sees live timer
- `explicit_confidence` - User provides confidence scores
- `revision_enabled` - Post-hoc revision allowed

### 4. Run the Server

```bash
python3 app.py
```

### 5. Open in Browser

Navigate to: **http://localhost:5000**

---

## 🎨 UI Features

### Main Interface

![UI Mockup Description]

**Top Section:**
- Session info (Participant ID, Comparison #, Condition badge)
- Condition-specific message (e.g., "Your response time will affect learning")
- Live timer (transparent condition only)

**Comparison Area:**
- Two trajectory panels side-by-side
- Video auto-plays on load
- Replay buttons (enabled after initial viewing)
- Replay counters per trajectory

**Preference Selection:**
- Clear A/B choice buttons
- Keyboard shortcuts (A/B keys)

**Confidence Rating:**
- 1-5 scale with labels
- "Very Unsure" → "Very Confident"
- Only shown if configured in condition

**Feedback:**
- Immediate accuracy feedback
- Timing summary
- Next comparison button

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` / `B` | Choose trajectory |
| `1-5` | Confidence score |
| `Space` / `Enter` | Next comparison |
| `Ctrl+S` | Session summary |
| `Ctrl+D` | Debug panel |

---

## 📊 Comprehensive Timing Tracking

### 5 Key Metrics

1. **Total Feedback Time** (entire interaction)
2. **Initial Viewing Time** (before first interaction)
3. **Deliberation Time** ⭐ PRIMARY (prompt → choice)
4. **Confidence Reporting Time** (choice → confidence)
5. **Replay Behavior** (count + duration)

See **TIMING_REFERENCE.md** for complete details.

### Example Timeline

```
Time     Event                      Metric
─────────────────────────────────────────────
0.0s     Trajectories shown        ← Total START
2.1s     First interaction         ← Initial viewing END
15.6s    Prompt shown             ← Deliberation START
27.9s    Preference selected (A)   ← Deliberation END
29.7s    Confidence submitted (4)  ← Total END

Results:
- Total: 29.7s
- Initial viewing: 2.1s
- Deliberation: 12.3s (PRIMARY METRIC)
- Confidence: 1.8s
```

---

## 📈 Data Output

### CSV Format

Saved to: `preference_data/P{id}_session{num}_{timestamp}_comparisons.csv`

**Key columns:**

| Column | Type | Description |
|--------|------|-------------|
| `comparison_id` | str | Unique ID (P01_S1_C001) |
| `choice_made` | str | A or B |
| `choice_accuracy` | bool | Correct vs ground truth |
| `confidence_score` | int | 1-5 (if applicable) |
| `deliberation_time` | float | **Primary metric** (seconds) |
| `total_feedback_time` | float | Full duration (seconds) |
| `replay_count` | int | Total replays |
| `traj_a_reward` | float | Ground truth reward A |
| `traj_b_reward` | float | Ground truth reward B |
| `difficulty_category` | str | Easy/Medium/Hard |

**53 total columns** including all timestamps and metadata.

### Example Analysis

```python
import pandas as pd

df = pd.read_csv('preference_data/P01_session1_..._comparisons.csv')

# Primary hypothesis test
correct = df[df['choice_accuracy'] == True]['deliberation_time']
incorrect = df[df['choice_accuracy'] == False]['deliberation_time']

print(f"Correct: {correct.mean():.2f}s (n={len(correct)})")
print(f"Incorrect: {incorrect.mean():.2f}s (n={len(incorrect)})")

# Expected if hypothesis is TRUE:
# Correct: 3.21s (n=380)
# Incorrect: 5.47s (n=107)
```

---

## 🔌 Integration with PEBBLE

### Load Trajectories from Reward Model

In your PEBBLE training code:

```python
from trajectory_video_generator import TrajectoryVideoGenerator, load_trajectories_from_pebble

# After reward model is instantiated
trajectory_pairs = load_trajectories_from_pebble(
    reward_model,
    num_comparisons=50,
    segment_length=50
)

# Generate videos
generator = TrajectoryVideoGenerator()
generator.generate_comparison_videos(
    trajectory_pairs,
    output_dir='preference_ui/static/videos'
)
generator.close()
```

### Load Human Preferences Back

```python
import pandas as pd

df = pd.read_csv('preference_data/P01_session1_..._comparisons.csv')

# Extract labels
labels = (df['choice_made'] == 'B').astype(int)  # 0=A, 1=B

# Optional: Weight by deliberation (inverse = faster → more confident)
weights = 1.0 / (1.0 + df['deliberation_time'])

# Put into PEBBLE
reward_model.put_queries(sa_t_1, sa_t_2, labels)
```

---

## 📱 Responsive Design

- ✅ Desktop (1400px+) - Full side-by-side layout
- ✅ Tablet (768-1024px) - Stacked layout
- ✅ Mobile (< 768px) - Optimized controls

---

## 🧪 Testing

### Debug Mode

Press `Ctrl+D` to toggle debug panel showing:
- Current comparison ID
- Condition
- Choice made
- Confidence score
- Replay counts
- Interaction state

### Session Summary

Press `Ctrl+S` to view:
- Total comparisons
- Accuracy rate
- Avg deliberation time
- Total replays
- Time statistics

---

## 🔧 Customization

### Change Confidence Scale (1-7 instead of 1-5)

1. Edit `templates/index.html`:
   ```html
   <button class="confidence-btn" data-value="6">6</button>
   <button class="confidence-btn" data-value="7">7</button>
   ```

2. Update labels in `static/css/style.css`

### Change Video Duration

Edit `trajectory_video_generator.py`:
```python
SEGMENT_LENGTH = 100  # timesteps (default: 50)
```

### Add Custom Instructions

Edit `static/js/app.js`, function `showConditionMessage()`:
```javascript
case 'baseline':
    message = 'Your custom instructions here...';
    break;
```

---

## 🐛 Troubleshooting

### Videos Don't Play

**Issue:** Browser blocks autoplay  
**Solution:** Serve over HTTPS or click play manually

### Timing Data Missing

**Issue:** Network errors  
**Solution:** Check browser console; verify server running

### CSV Not Created

**Issue:** File permissions  
**Solution:** Check write permissions on `preference_data/`

### MuJoCo/Gymnasium Errors

**Issue:** Environment not installed  
**Solution:** 
```bash
pip install "gymnasium[mujoco]" --break-system-packages
```

---

## 📄 Documentation

| File | Description |
|------|-------------|
| `README.md` | Full setup and usage guide |
| `TIMING_REFERENCE.md` | Complete timing system documentation |
| `app.py` | Inline code comments |
| `trajectory_video_generator.py` | Video generation docs |

---

## 🎓 For Your Thesis

### Data to Report

1. **Primary metric:** Deliberation time correlation with accuracy
2. **Secondary metrics:** Replay behavior, confidence calibration
3. **Timing breakdown:** Initial viewing, deliberation, confidence phases
4. **Condition comparisons:** Transparent vs. Opaque timing awareness

### Expected Results (Hypothesis TRUE)

```
Incorrect choices: M=5.47s, SD=2.31
Correct choices: M=3.21s, SD=1.42
t(487)=8.34, p<0.001, Cohen's d=1.13

"Longer deliberation times were significantly associated with 
incorrect preference selections..."
```

---

## 📞 Support

**Authors:**  
Yang Guo - University of Alberta  
Marzieh Ghayour - University of Alberta

**Issues?** Check:
1. README.md (comprehensive guide)
2. TIMING_REFERENCE.md (timing specifics)
3. Browser console (debugging)
4. Terminal output (server logs)

---

## ✅ Checklist

Before running your first session:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Trajectories generated (or videos placed in `static/videos/`)
- [ ] `app.py` configured with participant ID and condition
- [ ] Server running (`python3 app.py`)
- [ ] Browser open to `http://localhost:5000`
- [ ] CSV output directory writable (`preference_data/`)
- [ ] Participant briefed on task and condition

---

## 🚀 Next Steps

1. **Test with dummy data:**
   ```bash
   python3 trajectory_video_generator.py
   python3 app.py
   # Open http://localhost:5000
   ```

2. **Generate real trajectories:**
   - Modify `trajectory_video_generator.py` to load from PEBBLE
   - Or create videos from your existing trajectory data

3. **Configure conditions:**
   - Assign participants to different conditions
   - Randomize or counterbalance order

4. **Collect data:**
   - Run sessions
   - Monitor CSV output
   - Backup data regularly

5. **Analyze results:**
   - Use pandas to load CSV
   - Test primary hypothesis (deliberation vs. accuracy)
   - Generate figures for paper

---

**Everything is ready to use! Good luck with your experiment! 🎉**
