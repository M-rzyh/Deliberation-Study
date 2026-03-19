# Complete Step-by-Step User Guide
## PEBBLE Preference Collection UI for Walker2D

## PROJECT STRUCTURE OVERVIEW

Your research project has two main components:

```
Deliberation-Study/
│
├── PEBBLE/                          # RL training code
│   ├── train_PEBBLE.py             # Main training script
│   ├── config/                      # Hydra configs
│   │   ├── train_PEBBLE.yaml
│   │   └── agent/sac.yaml
│   ├── agent/                       # Agent implementations
│   │   ├── sac.py
│   │   ├── actor.py
│   │   └── critic.py
│   ├── custom_dmc2gym/              # Custom environment wrapper
│   ├── custom_dmcontrol/            # Custom DM Control
│   ├── logs/                        # Training logs & checkpoints
│   │   └── pebble/
│   │       ├── actor_1000.pt
│   │       ├── critic_1000.pt
│   │       └── reward_model_1000_*.pt
│   └── ...
│
└── preference_ui/                   # Human preference collection
    ├── app.py                       # Flask web server
    ├── trajectory_video_generator.py # Video generation
    ├── templates/                   # HTML templates
    │   └── index.html
    ├── static/                      # CSS, JS, videos
    │   ├── css/style.css
    │   ├── js/app.js
    │   └── videos/                  # Generated trajectory videos
    │       ├── walker_traj_a_1.mp4
    │       ├── walker_traj_b_1.mp4
    │       └── ...
    ├── preference_data/             # Collected preference data
    │   └── P01_session1_*.csv
    ├── requirements.txt
    ├── COMPLETE_USER_GUIDE.md       # This file
    └── ...
```

**Key Points:**
- **PEBBLE/** uses `pebble` conda environment (Python 3.8)
- **preference_ui/** uses `ui` conda environment (Python 3.10)
- They are **separate but connected** - PEBBLE trains the model, UI collects preferences

---

## TABLE OF CONTENTS

1. [Before You Start](#before-you-start)
2. [Installation (First Time Setup)](#installation-first-time-setup)
3. [Preparing Trajectory Videos](#preparing-trajectory-videos)
4. [Configuring for Each Participant](#configuring-for-each-participant)
5. [Running a Data Collection Session](#running-a-data-collection-session)
6. [Understanding the UI](#understanding-the-ui)
7. [Accessing Your Data](#accessing-your-data)
8. [Data Analysis](#data-analysis)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

---

## BEFORE YOU START

### What You Need

✅ **Computer with:**
- Python 3.8 or newer
- At least 2GB free disk space
- Web browser (Chrome, Firefox, Safari, or Edge)
- Internet connection (for installing packages)

✅ **Files you received:**
- `preference_ui_complete.zip` (download from outputs)

✅ **Knowledge needed:**
- Basic command line usage
- Basic text editing

### Time Required

- **First-time setup:** 15-30 minutes
- **Per participant session:** 30-60 minutes
- **Data analysis:** 10-20 minutes per session

---

## INSTALLATION (FIRST TIME SETUP)

### Step 0: Prerequisites - Set Up Conda Environments

**IMPORTANT:** You need TWO separate environments because PEBBLE and the UI have conflicting dependencies.

#### Environment 1: PEBBLE Training (Python 3.8)

```bash
# Create environment for PEBBLE
conda create -n pebble python=3.8 -y
conda activate pebble

# Install PyTorch
conda install pytorch torchvision -c pytorch -y

# Install PEBBLE dependencies
conda install numpy=1.23.5 -y
pip install tensorboard==2.10.0
pip install gym==0.23.1
pip install opencv-python
pip install hydra-core
pip install dm-control
pip install mujoco
pip install tqdm
pip install termcolor
pip install scikit-image
pip install omegaconf
pip install cloudpickle

# Install custom modules
cd ~/Downloads/Deliberation-Study/PEBBLE/custom_dmc2gym
pip install -e .
cd ../custom_dmcontrol
pip install -e .
cd ..
```

#### Environment 2: UI (Python 3.10)

```bash
# Create environment for UI
conda create -n ui python=3.10 -y
conda activate ui

# Install UI dependencies
pip install flask numpy pandas opencv-python gymnasium tqdm
```

**When to use which environment:**
- Use `pebble` environment: Training PEBBLE, generating trajectory videos
- Use `ui` environment: Running the preference collection web interface

---

### Step 1: Extract the Files

**Your project structure should look like this:**

```
Deliberation-Study/
├── PEBBLE/                    # PEBBLE training code (pebble env)
│   ├── train_PEBBLE.py
│   ├── config/
│   ├── agent/
│   ├── logs/
│   └── ...
│
└── preference_ui/             # UI for collecting preferences (ui env)
    ├── app.py
    ├── templates/
    ├── static/
    ├── preference_data/
    └── ...
```

**If you don't have this structure yet:**

**On Mac/Linux:**
```bash
cd ~/Downloads/Deliberation-Study
# Extract the UI zip if needed
unzip preference_ui_complete.zip
```

**On Windows:**
1. Right-click `preference_ui_complete.zip`
2. Select "Extract All..."
3. Choose `C:\Users\YourName\Documents\Deliberation-Study\`
4. Click "Extract"

### Step 2: Open Terminal/Command Prompt

**On Windows:**
1. Press `Windows + R`
2. Type `cmd` and press Enter
3. Activate the UI environment:
   ```
   conda activate ui
   ```
4. Navigate to the folder:
   ```
   cd C:\Users\YourName\Documents\Deliberation-Study\preference_ui
   ```

**On Mac:**
1. Press `Cmd + Space`
2. Type "Terminal" and press Enter
3. Activate the UI environment:
   ```bash
   conda activate ui
   ```
4. Navigate to the folder:
   ```bash
   cd ~/Downloads/Deliberation-Study/preference_ui
   ```

**On Linux:**
```bash
conda activate ui
cd ~/Downloads/Deliberation-Study/preference_ui
```

### Step 3: Verify Python Installation

**Make sure you're in the `ui` conda environment:**

```bash
# Should show (ui) at the start of your prompt
conda activate ui

# Check Python version
python3 --version
```

You should see something like: `Python 3.10.x` or newer

**If conda is not installed:**
- Download and install Miniconda or Anaconda from https://docs.conda.io/en/latest/miniconda.html

### Step 4: Install Required Packages (UI Environment)

**Make sure you're in the `ui` environment:**

```bash
conda activate ui
```

**Install packages:**

```bash
pip install flask numpy pandas opencv-python gymnasium tqdm
```

**Or use the requirements file:**

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Collecting flask...
Successfully installed flask-2.3.0 numpy-1.24.0 ...
```

**Note:** If you see errors, the conda environment should handle most dependency conflicts automatically.

### Step 5: Verify Installation

Run this test command:

```bash
python3 -c "import flask; print('Flask installed OK')"
```

Should print: `Flask installed OK`

### Step 6: Create Necessary Folders

**The setup script does this automatically:**

```bash
chmod +x setup.sh
./setup.sh
```

**Or create manually:**

```bash
mkdir -p static/videos
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
mkdir -p trajectory_data
mkdir -p preference_data
```

**Verify folder structure:**

```
preference_ui/
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── videos/          ← Videos go here
├── preference_data/      ← CSV logs saved here
└── trajectory_data/      ← Source trajectories
```

---

## PREPARING TRAJECTORY VIDEOS

You have **2 options** for getting trajectory videos:

### Option A: Generate Demo Videos (For Testing)

**Use this to test the UI before collecting real data.**

1. Run the generator:
   ```bash
   python3 trajectory_video_generator.py
   ```

2. You'll see:
   ```
   Generating 10 comparison pairs...
   ✓ Saved video: static/videos/walker_traj_a_1.mp4
   ✓ Saved video: static/videos/walker_traj_b_1.mp4
   ...
   ✓ Generated 10 comparison pairs
   ```

3. Check the videos:
   ```bash
   ls static/videos/
   ```
   
   Should show:
   ```
   walker_traj_a_1.mp4  walker_traj_b_1.mp4
   walker_traj_a_2.mp4  walker_traj_b_2.mp4
   ...
   ```

**Demo videos use random Walker2D policies** - they won't be meaningful comparisons, but perfect for testing the UI!

### Option B: Generate from Your PEBBLE Training

**Use this for real data collection.**

#### Step B1: Modify Your PEBBLE Code

**IMPORTANT: Use the `pebble` conda environment for this:**

```bash
conda activate pebble
cd ~/Downloads/Deliberation-Study-main
```

Add this to your `train_PEBBLE.py` or `reward_model.py`:

```python
# At the end of your training script
from trajectory_video_generator import TrajectoryVideoGenerator

def export_comparisons_for_ui(reward_model, output_dir='../preference_ui/static/videos', num_comparisons=50):
    """Export trajectory pairs as videos for UI"""
    
    # Get trajectory segments from reward model
    sa_t_1, sa_t_2, r_t_1, r_t_2 = reward_model.get_queries(mb_size=num_comparisons)
    
    trajectory_pairs = []
    
    for i in range(num_comparisons):
        # Extract states and actions
        segment_a = sa_t_1[i]  # shape: (segment_length, obs_dim + action_dim)
        segment_b = sa_t_2[i]
        
        # Split into states and actions
        states_a = segment_a[:, :reward_model.ds]  # first ds dimensions
        actions_a = segment_a[:, reward_model.ds:]  # rest are actions
        
        states_b = segment_b[:, :reward_model.ds]
        actions_b = segment_b[:, reward_model.ds:]
        
        pair = {
            'states_a': states_a,
            'actions_a': actions_a,
            'states_b': states_b,
            'actions_b': actions_b
        }
        
        trajectory_pairs.append(pair)
    
    # Generate videos
    generator = TrajectoryVideoGenerator(env_name='Walker2d-v4')
    generator.generate_comparison_videos(trajectory_pairs, output_dir=output_dir)
    generator.close()
    
    print(f"✓ Exported {num_comparisons} comparisons to {output_dir}")

# Call after training
export_comparisons_for_ui(workspace.reward_model, num_comparisons=50)
```

#### Step B2: Run Your PEBBLE Training

**In the `pebble` environment:**

```bash
conda activate pebble
cd ~/Downloads/Deliberation-Study-main
python3 train_PEBBLE.py
```

After training completes, videos will be in `preference_ui/static/videos/`

#### Step B3: Verify Videos

```bash
ls preference_ui/static/videos/ | head -10
```

Should show pairs of videos.

---

## CONFIGURING FOR EACH PARTICIPANT

**You need to configure the UI for each participant BEFORE they start.**

### Step 1: Open app.py

Use any text editor:
- Windows: Notepad, VS Code
- Mac: TextEdit, VS Code
- Linux: nano, vim, VS Code

```bash
# On Mac/Linux
nano app.py

# Or
code app.py  # if you have VS Code
```

### Step 2: Find the CONFIG Section

Look for lines 11-17:

```python
# Configuration
CONFIG = {
    'data_dir': 'trajectory_data',
    'output_dir': 'preference_data',
    'video_dir': 'static/videos',
    'participant_id': 'P01',        # ← CHANGE THIS
    'session_id': 1,                # ← CHANGE THIS
    'condition': 'baseline',        # ← CHANGE THIS
}
```

### Step 3: Set Participant ID

**Format:** `P01`, `P02`, `P03`, etc.

```python
'participant_id': 'P01',  # First participant
'participant_id': 'P02',  # Second participant
'participant_id': 'P03',  # Third participant
```

**Important:** Use consistent IDs! Track in a spreadsheet:

| Participant ID | Real Name | Date | Condition |
|----------------|-----------|------|-----------|
| P01 | Alice | 2026-03-18 | baseline |
| P02 | Bob | 2026-03-19 | time_aware_transparent |
| P03 | Carol | 2026-03-19 | explicit_confidence |

### Step 4: Set Session Number

For **each participant**, session IDs start at 1:

```python
'session_id': 1,  # First session with this participant
'session_id': 2,  # Second session with this participant (if doing multiple)
```

### Step 5: Set Experimental Condition

Choose ONE of these:

```python
'condition': 'baseline',
```

**Available conditions:**

| Condition | Description | What participant sees |
|-----------|-------------|----------------------|
| `baseline` | Standard PEBBLE | Standard instructions + confidence rating |
| `time_aware_opaque` | Time tracked silently | Standard instructions, NO timer, time logged |
| `time_aware_transparent` | User sees timer | **"Your response time affects learning"** + live timer |
| `explicit_confidence` | Confidence collected | Standard + confidence rating (1-5) |
| `revision_enabled` | Can revise later | Standard + "You can review choices later" |

**Recommendation for your study:**
- 5 participants per condition = 25 total participants
- Randomize condition assignment

### Step 6: Save the File

- **Notepad:** File → Save
- **VS Code:** Ctrl+S (Cmd+S on Mac)
- **nano:** Ctrl+X, then Y, then Enter

### Example Configurations

**Participant 1, Session 1, Baseline:**
```python
CONFIG = {
    'participant_id': 'P01',
    'session_id': 1,
    'condition': 'baseline',
}
```

**Participant 1, Session 2 (next day):**
```python
CONFIG = {
    'participant_id': 'P01',
    'session_id': 2,
    'condition': 'baseline',
}
```

**Participant 5, Session 1, Transparent condition:**
```python
CONFIG = {
    'participant_id': 'P05',
    'session_id': 1,
    'condition': 'time_aware_transparent',
}
```

---

## RUNNING A DATA COLLECTION SESSION

### Before the Participant Arrives

#### 0. Activate the UI Environment

**CRITICAL: Always use the `ui` environment for running the web server:**

```bash
conda activate ui
cd ~/Downloads/preference_ui
```

#### 1. Configure app.py (see previous section)

#### 2. Start the server

```bash
python3 app.py
```

**Expected output:**
```
==============================================================
PEBBLE Preference Collection UI
==============================================================
Participant ID: P01
Session ID: 1
Condition: baseline
Output: preference_data/P01_session1_20260318_143217_comparisons.csv
==============================================================

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

**✓ Success!** Server is running.

**If you see errors:**
- Port already in use: Kill other process or change port in app.py (line 468: `app.run(..., port=5001)`)
- Import errors: Re-run `pip install -r requirements.txt`

#### 3. Open browser and verify

Open: **http://localhost:5000**

You should see the comparison interface.

**Test it yourself:**
1. Watch both videos
2. Click "A" or "B"
3. Select confidence (if shown)
4. Click "Next Comparison"

**Verify CSV is being created:**
```bash
ls preference_data/
```

Should show: `P01_session1_TIMESTAMP_comparisons.csv`

#### 4. If everything works, STOP and restart fresh

Press `Ctrl+C` in terminal to stop server.

Delete test data:
```bash
rm preference_data/*.csv
```

Restart server:
```bash
python3 app.py
```

### When Participant Arrives

#### 1. Brief the participant

**Standard script:**

> "Thank you for participating! You'll be comparing pairs of robot walking behaviors. 
> 
> For each comparison:
> - Watch both videos (they play automatically)
> - Decide which robot walks better
> - Click A or B
> - [If applicable: Rate your confidence from 1-5]
> - Move to the next comparison
> 
> There are about 50 comparisons total, taking 20-30 minutes.
> 
> You can replay videos using the replay buttons if needed.
> 
> [If transparent condition: Your response time will be tracked and used to weight your feedback.]
> 
> Any questions?"

#### 2. Show them the interface

Open browser to: **http://localhost:5000**

Point out:
- Two video panels
- Replay buttons (available after first viewing)
- A/B choice buttons
- Confidence scale (if applicable)

#### 3. Let them do a practice comparison

**Important:** This will be logged, so mention it's practice but still counts.

Or: Use a separate session_id for practice (e.g., session_id=0)

#### 4. Leave them to complete the task

**Stay nearby** in case of technical issues, but don't hover.

**Typical duration:** 20-40 minutes for 50 comparisons

#### 5. Monitor progress (optional)

In a new terminal window:

```bash
# See how many comparisons completed
wc -l preference_data/P01_session1_*.csv

# Should show: number of rows = comparisons + 1 (header)
```

### After Participant Finishes

#### 1. Thank them

> "Thank you! You completed [X] comparisons. Your data has been saved."

#### 2. Stop the server

Press `Ctrl+C` in the terminal

#### 3. Backup the data IMMEDIATELY

```bash
# Copy to backup folder
cp preference_data/P01_session1_*.csv backups/

# Or create timestamped backup
cp preference_data/P01_session1_*.csv backups/P01_session1_backup_$(date +%Y%m%d_%H%M%S).csv
```

**CRITICAL:** Do this before running the next participant!

#### 4. Verify data integrity

Open the CSV in Excel or run:

```bash
python3 << 'EOF'
import pandas as pd
import glob

# Find the most recent CSV
files = glob.glob('preference_data/*.csv')
latest = max(files, key=lambda x: os.path.getctime(x))

df = pd.read_csv(latest)

print(f"File: {latest}")
print(f"Total rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Missing deliberation times: {df['deliberation_time'].isna().sum()}")
print(f"Accuracy rate: {df['choice_accuracy'].mean():.1%}")
EOF
```

Expected output:
```
File: preference_data/P01_session1_20260318_143217_comparisons.csv
Total rows: 50
Columns: 53
Missing deliberation times: 0
Accuracy rate: 78.0%
```

---

## UNDERSTANDING THE UI

### Main Interface Components

```
┌─────────────────────────────────────────────────────────┐
│  Which trajectory demonstrates better Walker2D behavior?│
│  Participant: P01    Comparison: 5    [baseline]        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │  Trajectory A     │      │  Trajectory B     │       │
│  │  [🔄 Replay]     │      │  [🔄 Replay]     │       │
│  │                   │      │                   │       │
│  │  [Video Player]   │      │  [Video Player]   │       │
│  │                   │      │                   │       │
│  │  Frames: 50       │      │  Frames: 50       │       │
│  │  Replays: 0       │      │  Replays: 1       │       │
│  └──────────────────┘      └──────────────────┘       │
│                                                          │
│           Which trajectory is better?                    │
│                                                          │
│         ┌─────────┐        ┌─────────┐                 │
│         │    A    │        │    B    │                 │
│         │Trajectory│        │Trajectory│                 │
│         │A is better        │B is better│                 │
│         └─────────┘        └─────────┘                 │
│                                                          │
│      How confident are you in your choice?              │
│                                                          │
│    1      2      3      4      5                        │
│ [Guessing] [Unsure] [Moderate] [Confident] [Certain]   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### What Happens During a Comparison

**Timeline:**

1. **0:00** - Trajectories load and auto-play
   - Both videos start simultaneously
   - Replay buttons are disabled
   - **Timing starts:** `trajectories_shown`

2. **~0:05** - Videos finish playing
   - Replay buttons become enabled
   - Participant can rewatch if needed

3. **User replays (optional)**
   - Click "Replay" button
   - Video restarts from beginning
   - **Timing logged:** `replay_a_start_N`, `replay_a_end_N`
   - Replay counter increments

4. **Both videos watched**
   - Preference prompt appears: "Which trajectory is better?"
   - **Timing logged:** `prompt_shown`

5. **User clicks A or B**
   - Button turns green
   - **Timing logged:** `preference_selected`
   - **Deliberation time calculated:** (preference_selected - prompt_shown)

6. **Confidence rating (if applicable)**
   - 1-5 scale appears
   - User clicks a number

7. **User submits confidence**
   - **Timing logged:** `confidence_submitted`
   - **Total feedback time calculated:** (confidence_submitted - trajectories_shown)

8. **Feedback shown**
   - ✓ or ✗ icon (based on accuracy)
   - "You have completed X comparisons"
   - Timing summary displayed

9. **Next comparison**
   - User clicks "Next Comparison" button
   - Process repeats

### Keyboard Shortcuts

| Key | Action | Notes |
|-----|--------|-------|
| `A` | Choose trajectory A | Only when prompt is shown |
| `B` | Choose trajectory B | Only when prompt is shown |
| `1-5` | Select confidence | Only when confidence prompt shown |
| `Space` | Next comparison | Only after feedback shown |
| `Enter` | Next comparison | Only after feedback shown |
| `Ctrl+S` | Session summary | Shows statistics anytime |
| `Ctrl+D` | Debug panel | Toggle debug info |

---

## ACCESSING YOUR DATA

### Where is the data?

```
preference_data/
└── P01_session1_20260318_143217_comparisons.csv
    └── Format: {participant_id}_session{session_id}_{timestamp}_comparisons.csv
```

### Opening the CSV

**Excel:**
1. Open Excel
2. File → Open
3. Navigate to `preference_data/`
4. Select the CSV file
5. Click "Open"

**Google Sheets:**
1. Go to sheets.google.com
2. File → Import → Upload
3. Select the CSV
4. Click "Import data"

**Python/Pandas:**
```python
import pandas as pd

df = pd.read_csv('preference_data/P01_session1_20260318_143217_comparisons.csv')
print(df.head())
```

### Understanding the Columns

**Most Important Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `comparison_id` | str | Unique ID | P01_S1_C001 |
| `comparison_number` | int | Sequential number | 1, 2, 3... |
| `choice_made` | str | User's choice | A or B |
| `choice_accuracy` | bool | Correct vs ground truth | True/False |
| `deliberation_time` | float | **PRIMARY METRIC** (seconds) | 3.542 |
| `total_feedback_time` | float | Full duration (seconds) | 25.318 |
| `confidence_score` | int | 1-5 rating | 4 |
| `traj_a_reward` | float | Ground truth reward A | 342.5 |
| `traj_b_reward` | float | Ground truth reward B | 287.3 |
| `correct_choice` | str | Ground truth answer | A or B |
| `difficulty_category` | str | Easy/Medium/Hard | Medium |
| `replay_count` | int | Total replays | 2 |

**All 53 columns documented in README.md**

### Quick Data Check

```bash
# Count comparisons
wc -l preference_data/P01_session1_*.csv

# View first 10 rows
head -10 preference_data/P01_session1_*.csv

# Check for missing values
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('preference_data/P01_session1_20260318_143217_comparisons.csv')
print(df.isnull().sum())
EOF
```

---

## DATA ANALYSIS

### Quick Analysis in Python

Create a file `analyze.py`:

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('preference_data/P01_session1_20260318_143217_comparisons.csv')

print(f"\n{'='*60}")
print(f"SESSION SUMMARY: {df['participant_id'].iloc[0]}, Session {df['session_id'].iloc[0]}")
print(f"{'='*60}\n")

# Basic stats
print(f"Total comparisons: {len(df)}")
print(f"Accuracy: {df['choice_accuracy'].mean():.1%}")
print(f"Avg deliberation: {df['deliberation_time'].mean():.2f}s")
print(f"Replays: {df['replay_count'].sum()} total ({(df['replay_count'] > 0).mean():.1%} of comparisons)")

# Primary hypothesis test
correct = df[df['choice_accuracy'] == True]['deliberation_time']
incorrect = df[df['choice_accuracy'] == False]['deliberation_time']

print(f"\n{'='*60}")
print("PRIMARY HYPOTHESIS: Deliberation Time vs. Accuracy")
print(f"{'='*60}")
print(f"Correct (n={len(correct)}): M={correct.mean():.2f}s, SD={correct.std():.2f}s")
print(f"Incorrect (n={len(incorrect)}): M={incorrect.mean():.2f}s, SD={incorrect.std():.2f}s")

if len(correct) > 0 and len(incorrect) > 0:
    t_stat, p_value = stats.ttest_ind(incorrect, correct)
    print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ Significant difference (p < 0.05)")
        print(f"Incorrect choices took {incorrect.mean() - correct.mean():.2f}s longer on average")
    else:
        print("No significant difference (p >= 0.05)")
else:
    print("\nNot enough data for t-test")

# By difficulty
print(f"\n{'='*60}")
print("BY DIFFICULTY")
print(f"{'='*60}")
for diff in ['Easy', 'Medium', 'Hard']:
    subset = df[df['difficulty_category'] == diff]
    if len(subset) > 0:
        print(f"{diff}: n={len(subset)}, accuracy={subset['choice_accuracy'].mean():.1%}, "
              f"delib={subset['deliberation_time'].mean():.2f}s")
```

Run it:
```bash
python3 analyze.py
```

### Expected Output

```
============================================================
SESSION SUMMARY: P01, Session 1
============================================================

Total comparisons: 50
Accuracy: 78.0%
Avg deliberation: 4.23s
Replays: 12 total (24.0% of comparisons)

============================================================
PRIMARY HYPOTHESIS: Deliberation Time vs. Accuracy
============================================================
Correct (n=39): M=3.21s, SD=1.42s
Incorrect (n=11): M=5.47s, SD=2.31s

t-test: t=3.421, p=0.0012
✓ Significant difference (p < 0.05)
Incorrect choices took 2.26s longer on average

============================================================
BY DIFFICULTY
============================================================
Easy: n=18, accuracy=94.4%, delib=2.85s
Medium: n=21, accuracy=76.2%, delib=4.12s
Hard: n=11, accuracy=54.5%, delib=6.73s
```

### Combining Multiple Sessions

```python
import pandas as pd
import glob

# Load all CSV files
files = glob.glob('preference_data/*.csv')
dfs = [pd.read_csv(f) for f in files]

# Combine
all_data = pd.concat(dfs, ignore_index=True)

print(f"Total participants: {all_data['participant_id'].nunique()}")
print(f"Total comparisons: {len(all_data)}")
print(f"Overall accuracy: {all_data['choice_accuracy'].mean():.1%}")

# By condition
print("\nBy Condition:")
for condition in all_data['condition'].unique():
    subset = all_data[all_data['condition'] == condition]
    print(f"{condition}: n={len(subset)}, accuracy={subset['choice_accuracy'].mean():.1%}")
```

---

## TROUBLESHOOTING

### Problem: Server won't start

**Error:** `Address already in use`

**Solution:**
```bash
# Find and kill the process using port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port
# Edit app.py, line 468: app.run(..., port=5001)
```

### Problem: Videos don't play

**Error:** Black screen or "cannot play video"

**Possible causes:**

1. **Videos not generated:**
   ```bash
   ls static/videos/
   # Should show .mp4 files
   ```
   **Fix:** Run `python3 trajectory_video_generator.py`

2. **Wrong video codec:**
   Edit `trajectory_video_generator.py`, line 63:
   ```python
   fourcc='avc1'  # Try instead of 'mp4v'
   ```

3. **Browser blocks autoplay:**
   **Fix:** Click the play button manually, or use Chrome/Firefox

### Problem: Timing data is missing

**Symptom:** `deliberation_time` column is empty (NaN)

**Cause:** JavaScript errors or network issues

**Fix:**
1. Open browser console (F12)
2. Look for red errors
3. Check `/api/mark_timing` calls are succeeding

### Problem: CSV not created

**Symptom:** `preference_data/` folder is empty

**Cause:** File permission issues

**Fix:**
```bash
# Check permissions
ls -la preference_data/

# Fix permissions
chmod 755 preference_data/
```

### Problem: Data looks wrong

**Symptom:** All deliberation times are 0.0s or very short

**Cause:** User is clicking too fast (not actually watching)

**Fix:**
- Remind participant to watch both videos fully
- Check if videos are actually playing (codec issue)
- Consider adding minimum deliberation time validation

### Problem: Import errors

**Error:** `ModuleNotFoundError: No module named 'flask'`

**Fix:**
```bash
pip install flask --break-system-packages

# Or reinstall all
pip install -r requirements.txt --break-system-packages
```

### Problem: Gymnasium/MuJoCo errors

**Error:** `No module named 'mujoco'`

**Fix:**
```bash
pip install "gymnasium[mujoco]" --break-system-packages
```

**Still broken?** Try:
```bash
pip install mujoco==2.3.7 --break-system-packages
```

---

## ADVANCED FEATURES

### Custom Number of Comparisons

By default, UI loads comparisons infinitely until you stop.

To limit:

1. Edit `app.py`, add after line 81:
   ```python
   max_comparisons = 50  # Set your limit
   ```

2. In `get_comparison()` function, add:
   ```python
   if current_comparison['comparison_number'] >= max_comparisons:
       return jsonify({'status': 'complete', 'message': 'Session complete!'})
   ```

### Adding Custom Metadata

To log additional info per participant:

1. Edit `app.py`, add to CONFIG:
   ```python
   'participant_age': 25,
   'participant_gender': 'F',
   'participant_experience': 'novice',
   ```

2. In `submit_preference()`, add to `log_entry`:
   ```python
   'participant_age': CONFIG['participant_age'],
   'participant_gender': CONFIG['participant_gender'],
   ```

3. Update `ComparisonLogger.headers` to include new columns

### Enabling Dark Mode

Edit `static/css/style.css`, add:

```css
body {
    background: #1a1a1a;
    color: #e0e0e0;
}

.container {
    background: #2d2d2d;
}

.trajectory-panel {
    background: #3a3a3a;
}
```

### Real-time Monitoring Dashboard

Create `monitor.py`:

```python
import pandas as pd
import time
import os

print("Monitoring active session...")
print("Press Ctrl+C to stop\n")

last_count = 0

while True:
    try:
        files = [f for f in os.listdir('preference_data') if f.endswith('.csv')]
        if files:
            latest = max([f'preference_data/{f}' for f in files], 
                        key=os.path.getctime)
            df = pd.read_csv(latest)
            
            if len(df) > last_count:
                last_count = len(df)
                accuracy = df['choice_accuracy'].mean()
                avg_delib = df['deliberation_time'].mean()
                
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Comparisons: {len(df)}, "
                      f"Accuracy: {accuracy:.1%}, "
                      f"Avg deliberation: {avg_delib:.2f}s")
        
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        break
```

Run in separate terminal:
```bash
python3 monitor.py
```

---

## FINAL CHECKLIST

### Before Each Session:

- [ ] `app.py` configured with correct participant_id
- [ ] `app.py` configured with correct session_id
- [ ] `app.py` configured with correct condition
- [ ] Videos generated and in `static/videos/`
- [ ] Server starts without errors
- [ ] Browser opens to http://localhost:5000
- [ ] Test comparison works (watch, choose, confidence, next)
- [ ] CSV is created in `preference_data/`
- [ ] Previous session data backed up

### After Each Session:

- [ ] Thank participant
- [ ] Stop server (Ctrl+C)
- [ ] Backup CSV immediately
- [ ] Verify data integrity (run quick check)
- [ ] Update participant tracking spreadsheet
- [ ] Prepare for next participant (update CONFIG)

### Data Collection Complete:

- [ ] All participant CSVs backed up
- [ ] Combined dataset created
- [ ] Preliminary analysis run
- [ ] Data uploaded to secure storage
- [ ] Paper/thesis figures generated

---

## WORKFLOW: PEBBLE → UI → PEBBLE

### Complete Integration Workflow

**Phase 1: Train PEBBLE (pebble environment)**

```bash
# Activate PEBBLE environment
conda activate pebble
cd ~/Downloads/Deliberation-Study/PEBBLE

# Train PEBBLE
python3 train_PEBBLE.py num_train_steps=100000

# Export trajectories for UI (optional - add export function later)
# python3 export_for_ui.py
```

**Phase 2: Collect Human Preferences (ui environment)**

```bash
# Switch to UI environment
conda deactivate
conda activate ui
cd ~/Downloads/Deliberation-Study/preference_ui

# Generate demo videos for testing
python3 trajectory_video_generator.py

# Configure participant
nano app.py  # Set participant_id, session_id, condition

# Start server
python3 app.py

# Open http://localhost:5001
# Participant completes 50 comparisons
# Data saved to preference_data/P01_session1_*.csv

# Stop server (Ctrl+C)
# Backup data
cp preference_data/*.csv backups/
```

**Phase 3: Load Preferences Back (pebble environment)**

```bash
# Switch back to PEBBLE environment
conda deactivate
conda activate pebble
cd ~/Downloads/Deliberation-Study/PEBBLE

# Load human preferences (integration code to be added)
# python3 load_human_preferences.py

# Continue training with human feedback
# python3 train_PEBBLE.py load_preferences=true
```

---

## QUICK REFERENCE COMMANDS

```bash
# === UI ENVIRONMENT ===
conda activate ui

# Start server
python3 app.py

# Generate demo videos  
python3 trajectory_video_generator.py

# Check data
wc -l preference_data/*.csv

# Backup data
cp preference_data/*.csv backups/

# Quick analysis
python3 analyze.py

# === PEBBLE ENVIRONMENT ===
conda activate pebble

# Train PEBBLE
python3 train_PEBBLE.py

# Export trajectories
python3 export_for_ui.py

# Monitor live session
python3 monitor.py
```

---

## SUPPORT

**Check these resources first:**
1. /preference_ui/README.md - Comprehensive documentation
2. /preference_ui/TIMING_REFERENCE.md - Timing system details
3. /preference_ui/DEPLOYMENT_SUMMARY.md - Quick start guide

**Still stuck?**
- Check browser console (F12) for errors
- Check terminal for server errors
- Verify file permissions
- Re-run setup script

**Good luck with your data collection! 🚀**
