# FUNCTION REFERENCE & DATA FLOW GUIDE

## SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                     USER'S BROWSER                          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  index.html (UI Template)                          │   │
│  │  - Trajectory video players                        │   │
│  │  - Replay buttons                                  │   │
│  │  - Preference selection (A/B)                      │   │
│  │  - Confidence rating (1-5)                         │   │
│  └────────────────────────────────────────────────────┘   │
│         │                                   ▲                │
│         │ User interactions                 │ Display        │
│         ▼                                   │                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  app.js (Frontend Logic)                           │   │
│  │  - Track all timing events                         │   │
│  │  - Handle video playback                           │   │
│  │  - Send API calls to backend                       │   │
│  │  - Update UI based on responses                    │   │
│  └────────────────────────────────────────────────────┘   │
│         │                                   ▲                │
│         │ HTTP POST/GET                     │ JSON          │
│         ▼                                   │                │
└─────────────────────────────────────────────────────────────┘
         │                                   ▲
         │ Network                           │
         ▼                                   │
┌─────────────────────────────────────────────────────────────┐
│                   FLASK SERVER (app.py)                     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  API Endpoints                                      │   │
│  │  /api/get_comparison     - Load next trajectories  │   │
│  │  /api/mark_timing        - Log timing event        │   │
│  │  /api/submit_preference  - Submit choice+confidence│   │
│  │  /api/session_summary    - Get statistics          │   │
│  └────────────────────────────────────────────────────┘   │
│         │                                   │                │
│         ▼                                   ▼                │
│  ┌──────────────────┐            ┌──────────────────┐      │
│  │ TimingTracker    │            │ ComparisonLogger │      │
│  │ - Mark events    │            │ - Write to CSV   │      │
│  │ - Calculate Δt   │            │ - Log metadata   │      │
│  └──────────────────┘            └──────────────────┘      │
│                                            │                 │
│                                            ▼                 │
└─────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                              ┌──────────────────────────┐
                              │  preference_data/        │
                              │  P01_S1_..._comparisons.csv
                              └──────────────────────────┘
```

---

## KEY FUNCTIONS EXPLAINED

### BACKEND (app.py)

#### `class TimingTracker`
**Purpose:** Track all timing events for one comparison

```python
tracker = TimingTracker("P01_S1_C001")

# Mark events
tracker.mark('trajectories_shown')      # When videos start
tracker.mark('first_interaction')        # First mouse/keyboard event
tracker.mark('prompt_shown')             # When "Which is better?" appears
tracker.mark('preference_selected')      # When user clicks A or B
tracker.mark('confidence_submitted')     # When user submits confidence

# Calculate durations
durations = tracker.get_all_durations()
# Returns:
# {
#   'total_feedback_time': 29.718,      # Full duration
#   'initial_viewing_time': 2.118,      # Before interaction
#   'deliberation_time': 12.233,        # PRIMARY METRIC
#   'confidence_reporting_time': 1.851, # Confidence phase
#   'replay_count': 2,
#   'total_replay_time': 10.3
# }
```

**Key Methods:**

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `mark(event)` | Event name (str) | None | Record timestamp |
| `duration(start, end)` | Event names | Seconds (float) | Calculate time between events |
| `get_all_durations()` | None | Dict | All calculated durations |
| `to_dict()` | None | Dict | Export all data |

---

#### `class ComparisonLogger`
**Purpose:** Write comparison data to CSV

```python
logger = ComparisonLogger(
    output_dir='preference_data',
    participant_id='P01',
    session_id=1
)

# Log a comparison
log_entry = {
    'comparison_id': 'P01_S1_C001',
    'choice_made': 'A',
    'choice_accuracy': True,
    'deliberation_time': 3.542,
    'confidence_score': 4,
    # ... all other fields
}

logger.log_comparison(log_entry)
# Appends row to CSV
```

**CSV Columns (53 total):**

**Identifiers (7):**
- comparison_id, participant_id, session_id, comparison_number
- timestamp, condition, traj_a_id, traj_b_id

**Timing (10):**
- total_feedback_time
- initial_viewing_time
- deliberation_time ← **PRIMARY**
- confidence_reporting_time
- total_replay_time
- replay_count, replay_trajectory_a, replay_trajectory_b
- trajectories_shown_time, first_interaction_time
- (+ 5 more timestamp columns)

**Preference (4):**
- choice_made, choice_accuracy
- confidence_score, confidence_method

**Ground Truth (4):**
- traj_a_reward, traj_b_reward
- reward_difference, difficulty_category
- correct_choice

---

#### `@app.route('/api/get_comparison')`
**Purpose:** Fetch next trajectory comparison

**Flow:**
```
1. Increment comparison number
2. Generate comparison_id (e.g., "P01_S1_C001")
3. Initialize TimingTracker
4. Load trajectory data (videos, rewards)
5. Calculate ground truth (correct choice, difficulty)
6. Return JSON to frontend
```

**Response JSON:**
```json
{
  "comparison_id": "P01_S1_C001",
  "comparison_number": 1,
  "trajectory_a": {
    "id": "traj_a_1",
    "video_path": "/static/videos/walker_traj_a_1.mp4",
    "reward": 342.5,
    "length": 50
  },
  "trajectory_b": { ... },
  "condition": "baseline",
  "show_confidence": true,
  "show_timer": false
}
```

---

#### `@app.route('/api/mark_timing', methods=['POST'])`
**Purpose:** Log a timing event

**Request:**
```json
{
  "event": "preference_selected"
}
```

**Logged Events:**

| Event Name | When It Happens |
|------------|-----------------|
| `trajectories_shown` | Videos start playing |
| `first_interaction` | First mouse move/click |
| `prompt_shown` | "Which is better?" appears |
| `preference_selected` | User clicks A or B |
| `confidence_submitted` | User submits confidence |
| `replay_a_start_N` | Replay A begins (N = count) |
| `replay_a_end_N` | Replay A ends |
| `replay_b_start_N` | Replay B begins |
| `replay_b_end_N` | Replay B ends |

**Example sequence:**
```
00.000s → trajectories_shown
02.118s → first_interaction
15.617s → prompt_shown
18.850s → replay_a_start_1
23.950s → replay_a_end_1
27.867s → preference_selected
29.718s → confidence_submitted
```

---

#### `@app.route('/api/submit_preference', methods=['POST'])`
**Purpose:** Submit final preference + confidence

**Request:**
```json
{
  "choice": "A",
  "confidence": 4,
  "confidence_method": "explicit"
}
```

**What it does:**
1. Calculate `choice_accuracy` (compare to ground truth)
2. Get all timing durations from TimingTracker
3. Compile full log entry (53 fields)
4. Write to CSV via ComparisonLogger
5. Return feedback to user

**Response:**
```json
{
  "status": "success",
  "comparison_id": "P01_S1_C001",
  "accuracy": true,
  "total_comparisons": 1,
  "timing_summary": {
    "total_feedback_time": 29.718,
    "deliberation_time": 12.233,
    "replay_count": 1
  }
}
```

---

### FRONTEND (app.js)

#### `loadNextComparison()`
**Purpose:** Load a new comparison from server

**Flow:**
```javascript
1. Reset state (choice, confidence, replays)
2. Show loading screen
3. Fetch /api/get_comparison
4. Update UI (comparison number, videos)
5. Load videos into <video> elements
6. Mark timing: trajectories_shown
7. Start timer (if transparent condition)
8. Hide loading, show comparison
```

**Key variables updated:**
- `APP_STATE.currentComparison`
- `APP_STATE.choiceMade` → null
- `APP_STATE.confidenceScore` → null
- `APP_STATE.replayCountA` → 0
- `APP_STATE.replayCountB` → 0

---

#### `loadTrajectories(data)`
**Purpose:** Load and play both videos

```javascript
async function loadTrajectories(data) {
  const videoA = document.getElementById('video-a');
  const videoB = document.getElementById('video-b');
  
  // Set sources
  videoA.src = data.trajectory_a.video_path;
  videoB.src = data.trajectory_b.video_path;
  
  // Wait for both to load
  await Promise.all([...]);
  
  // Auto-play
  await videoA.play();
  await videoB.play();
}
```

**Events triggered:**
- `loadeddata` → onVideoLoaded()
- `ended` → onVideoEnded()

---

#### `onVideoEnded(trajectory)`
**Purpose:** Handle video finishing

```javascript
function onVideoEnded(trajectory) {
  // Check if BOTH videos finished
  if (videoA.ended && videoB.ended) {
    showPreferencePrompt();  // Show "Which is better?"
  }
}
```

**This triggers:**
- Preference buttons appear
- `mark_timing('prompt_shown')` called
- Deliberation period begins

---

#### `replayTrajectory(trajectory)`
**Purpose:** Replay a single video

**Flow:**
```javascript
1. Increment replay counter (APP_STATE.replayCountA++)
2. Update UI counter display
3. Mark timing: replay_a_start_N
4. Disable replay button
5. Reset video: video.currentTime = 0
6. Play video
7. On 'ended' event:
   - Re-enable button
   - Mark timing: replay_a_end_N
```

**Replay tracking:**
- Separate counters for A and B
- Each replay logged with unique event name
- Total replay time calculated from all start/end pairs

---

#### `makeChoice(choice)`
**Purpose:** User selected A or B

**Flow:**
```javascript
1. Store choice: APP_STATE.choiceMade = 'A'
2. Mark timing: preference_selected
3. Visual feedback (button turns green)
4. Disable both buttons
5. Hide preference section
6. Check if confidence needed:
   - If yes → showConfidencePrompt()
   - If no → submitPreference(null, null)
```

**Conditions that show confidence:**
- `condition === 'explicit_confidence'`
- `condition === 'baseline'`

---

#### `submitConfidence(score)`
**Purpose:** User selected confidence (1-5)

```javascript
1. Store score: APP_STATE.confidenceScore = 4
2. Visual feedback (button turns green)
3. Disable all confidence buttons
4. Call submitPreference(score, 'explicit')
```

---

#### `submitPreference(confidence, method)`
**Purpose:** Send final data to server

**Flow:**
```javascript
1. Mark timing: confidence_submitted (FINAL event)
2. Stop timer
3. POST to /api/submit_preference:
   {
     choice: APP_STATE.choiceMade,
     confidence: confidence,
     confidence_method: method
   }
4. Receive response (accuracy, timing summary)
5. showFeedback(result)
```

**Backend processes:**
- Calculates all durations
- Writes to CSV
- Returns accuracy and stats

---

#### `markTiming(event)`
**Purpose:** Send timing event to server

```javascript
async function markTiming(event) {
  await fetch('/api/mark_timing', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event: event })
  });
}
```

**Called throughout UI interaction:**
- On video load
- On first interaction
- On replay start/end
- On preference selection
- On confidence submission

---

#### `markFirstInteraction()`
**Purpose:** Detect user's first action

```javascript
// Set up once on page load
document.addEventListener('mousemove', markFirstInteraction, { once: true });
document.addEventListener('click', markFirstInteraction, { once: true });

function markFirstInteraction() {
  if (!APP_STATE.firstInteraction) {
    APP_STATE.firstInteraction = true;
    markTiming('first_interaction');
  }
}
```

**Why track this?**
- Measures "passive viewing" period
- Initial viewing time = first_interaction - trajectories_shown
- Indicates how long user watches before interacting

---

#### `startTimer()` (Transparent condition only)
**Purpose:** Show live elapsed time

```javascript
function startTimer() {
  APP_STATE.startTime = Date.now();
  
  APP_STATE.timerInterval = setInterval(() => {
    const elapsed = (Date.now() - APP_STATE.startTime) / 1000;
    document.getElementById('elapsed-time').textContent = 
      elapsed.toFixed(1) + 's';
  }, 100);  // Update every 100ms
}
```

**Visual display:**
```
┌─────────────────────────────┐
│ Time on this comparison:    │
│        15.3s                │
│ Your response time will be  │
│ used to weight your feedback│
└─────────────────────────────┘
```

---

## DATA FLOW DIAGRAM

### Complete Comparison Flow

```
USER ACTION                 FRONTEND                BACKEND                CSV
═══════════════════════════════════════════════════════════════════════════

[Page loads]
                         →  loadNextComparison()
                                    ↓
                            GET /api/get_comparison
                                    ↓               ↓
                                                Initialize
                                              TimingTracker
                                                    ↓
                            ← Return JSON
                                    ↓
                            Load videos
                                    ↓
                         →  POST mark_timing
                            event: trajectories_shown
                                                    ↓
                                              tracker.mark()

[Videos play]
                                    ↓
[User moves mouse]
                         →  POST mark_timing
                            event: first_interaction
                                                    ↓
                                              tracker.mark()

[Videos end]
                         →  showPreferencePrompt()
                         →  POST mark_timing
                            event: prompt_shown
                                                    ↓
                                              tracker.mark()

[User clicks "Replay A"]
                         →  replayTrajectory('A')
                         →  POST mark_timing
                            event: replay_a_start_1
                                                    ↓
                                              tracker.mark()
[Replay ends]
                         →  POST mark_timing
                            event: replay_a_end_1
                                                    ↓
                                              tracker.mark()

[User clicks "A"]
                         →  makeChoice('A')
                         →  POST mark_timing
                            event: preference_selected
                                                    ↓
                                              tracker.mark()

[User clicks "4"]
                         →  submitConfidence(4)
                         →  POST mark_timing
                            event: confidence_submitted
                                                    ↓
                                              tracker.mark()
                                    ↓
                            POST /api/submit_preference
                            { choice: 'A', confidence: 4 }
                                                    ↓
                                              Calculate accuracy
                                              Get durations
                                              Compile log_entry
                                                    ↓
                                              logger.log_comparison()
                                                                    ↓
                                                                Write row
                                                                to CSV
                            ← Return feedback
                                    ↓
                            showFeedback()

[User clicks "Next"]
                         →  loadNextComparison()
                                    ↓
                            [Repeat cycle]
```

---

## TIMING CALCULATION REFERENCE

### How Each Metric is Calculated

```python
# In TimingTracker.get_all_durations()

# 1. Total Feedback Time
total_feedback_time = (
    datetime.fromisoformat(events['confidence_submitted']) -
    datetime.fromisoformat(events['trajectories_shown'])
).total_seconds()

# 2. Initial Viewing Time
initial_viewing_time = (
    datetime.fromisoformat(events['first_interaction']) -
    datetime.fromisoformat(events['trajectories_shown'])
).total_seconds()

# 3. Deliberation Time (PRIMARY METRIC)
deliberation_time = (
    datetime.fromisoformat(events['preference_selected']) -
    datetime.fromisoformat(events['prompt_shown'])
).total_seconds()

# 4. Confidence Reporting Time
confidence_reporting_time = (
    datetime.fromisoformat(events['confidence_submitted']) -
    datetime.fromisoformat(events['preference_selected'])
).total_seconds()

# 5. Total Replay Time
replay_starts = [v for k, v in events.items() if 'replay_start' in k]
replay_ends = [v for k, v in events.items() if 'replay_end' in k]

if replay_starts and replay_ends:
    total_replay_time = (
        datetime.fromisoformat(max(replay_ends)) -
        datetime.fromisoformat(min(replay_starts))
    ).total_seconds()

# 6. Replay Count
replay_count = sum(1 for key in events if 'replay_' in key and 'start' in key)
```

---

## FILE REFERENCE

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `app.py` | 468 | Backend server | TimingTracker, ComparisonLogger, API routes |
| `app.js` | 450 | Frontend logic | loadNextComparison, makeChoice, submitPreference |
| `index.html` | 150 | UI template | Video players, buttons, forms |
| `style.css` | 600 | Styling | Responsive layout, animations |
| `trajectory_video_generator.py` | 280 | Video creation | render_trajectory, save_video |

---

## QUICK REFERENCE: WHAT HAPPENS WHEN

| User Action | Frontend Function | Backend Endpoint | What Gets Logged |
|-------------|-------------------|------------------|------------------|
| **Page loads** | `loadNextComparison()` | `GET /api/get_comparison` | New comparison created |
| **Videos start** | `loadTrajectories()` | `POST /api/mark_timing` | `trajectories_shown` |
| **Mouse moves** | `markFirstInteraction()` | `POST /api/mark_timing` | `first_interaction` |
| **Videos end** | `onVideoEnded()` | `POST /api/mark_timing` | `prompt_shown` |
| **Clicks Replay** | `replayTrajectory()` | `POST /api/mark_timing` | `replay_a_start_N`, `replay_a_end_N` |
| **Clicks A/B** | `makeChoice()` | `POST /api/mark_timing` | `preference_selected` |
| **Clicks 1-5** | `submitConfidence()` | `POST /api/mark_timing` | `confidence_submitted` |
| **Submits all** | `submitPreference()` | `POST /api/submit_preference` | Full row written to CSV |

---

**This reference covers all major functions and data flows in the system!**
