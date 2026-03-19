# 🚀 QUICK START CARD
**PEBBLE Preference Collection UI - Get Started in 5 Minutes**

---

## 📥 DOWNLOAD & EXTRACT

**Download:** `preference_ui_complete.zip` (46 KB)

```bash
unzip preference_ui_complete.zip
cd preference_ui
```

---

## 🔧 INSTALL (One-time)

```bash
pip install -r requirements.txt --break-system-packages
```

---

## 🎬 GENERATE DEMO VIDEOS (For testing)

```bash
python3 trajectory_video_generator.py
```

Creates 10 demo comparison pairs in `static/videos/`

---

## ⚙️ CONFIGURE PARTICIPANT

Edit `app.py`, lines 11-17:

```python
CONFIG = {
    'participant_id': 'P01',        # ← Change for each participant
    'session_id': 1,                # ← Increment per session
    'condition': 'baseline',        # ← See options below
}
```

**Conditions:**
- `baseline` - Standard + confidence
- `time_aware_opaque` - Time tracked silently
- `time_aware_transparent` - Shows live timer
- `explicit_confidence` - Confidence scores
- `revision_enabled` - Can revise later

---

## ▶️ START SERVER

```bash
python3 app.py
```

Expected output:
```
==============================================================
PEBBLE Preference Collection UI
==============================================================
Participant ID: P01
Session ID: 1
Condition: baseline
Output: preference_data/P01_session1_20260318_143217_comparisons.csv
==============================================================

 * Running on http://0.0.0.0:5000
```

---

## 🌐 OPEN BROWSER

Navigate to: **http://localhost:5000**

---

## 👤 PARTICIPANT INSTRUCTIONS

> "Watch both robot walking videos, then choose which one walks better (A or B).
> Rate your confidence (1-5). Replay videos if needed. About 50 comparisons, 20-30 minutes."

---

## 📊 YOUR DATA

Saved to: `preference_data/P01_session1_TIMESTAMP_comparisons.csv`

**Key columns:**
- `deliberation_time` ← PRIMARY METRIC (seconds)
- `choice_accuracy` ← Correct vs ground truth
- `confidence_score` ← 1-5 rating
- `replay_count` ← Number of replays

**53 total columns** with comprehensive timing and metadata

---

## 📈 QUICK ANALYSIS

```python
import pandas as pd

df = pd.read_csv('preference_data/P01_session1_..._comparisons.csv')

# Basic stats
print(f"Comparisons: {len(df)}")
print(f"Accuracy: {df['choice_accuracy'].mean():.1%}")
print(f"Avg deliberation: {df['deliberation_time'].mean():.2f}s")

# Hypothesis test
correct = df[df['choice_accuracy'] == True]['deliberation_time']
incorrect = df[df['choice_accuracy'] == False]['deliberation_time']

from scipy import stats
t_stat, p = stats.ttest_ind(incorrect, correct)
print(f"\nDeliberation: Correct={correct.mean():.2f}s, Incorrect={incorrect.mean():.2f}s")
print(f"t-test: p={p:.4f}")
```

---

## 📚 FULL DOCUMENTATION

| File | Purpose |
|------|---------|
| **COMPLETE_USER_GUIDE.md** | ← **START HERE** - Step-by-step instructions |
| **FUNCTION_REFERENCE.md** | Technical details & data flow |
| **TIMING_REFERENCE.md** | Timing system explained |
| **README.md** | Comprehensive documentation |
| **DEPLOYMENT_SUMMARY.md** | Quick deployment guide |

---

## ⌨️ KEYBOARD SHORTCUTS

| Key | Action |
|-----|--------|
| `A` / `B` | Choose trajectory |
| `1-5` | Confidence score |
| `Space` | Next comparison |
| `Ctrl+S` | Session summary |
| `Ctrl+D` | Debug panel |

---

## 🔥 WHAT'S TRACKED

### 5 Timing Metrics:

1. **Total Feedback Time** - Full duration
2. **Initial Viewing Time** - Before first interaction
3. **Deliberation Time** ⭐ **PRIMARY** - Prompt → choice
4. **Confidence Reporting Time** - Choice → confidence
5. **Replay Behavior** - Count + duration

---

## ✅ SESSION CHECKLIST

**Before:**
- [ ] Videos generated (`ls static/videos/`)
- [ ] CONFIG set in app.py
- [ ] Server running (`python3 app.py`)
- [ ] Browser opens to localhost:5000
- [ ] Test comparison works

**After:**
- [ ] Stop server (Ctrl+C)
- [ ] Backup CSV: `cp preference_data/*.csv backups/`
- [ ] Verify data: Check CSV has ~50 rows
- [ ] Update CONFIG for next participant

---

## 🐛 TROUBLESHOOTING

**Server won't start:**
```bash
lsof -ti:5000 | xargs kill -9  # Kill existing process
```

**Videos won't play:**
```bash
ls static/videos/  # Check videos exist
# Try different codec in trajectory_video_generator.py
```

**Import errors:**
```bash
pip install flask numpy pandas opencv-python gymnasium --break-system-packages
```

---

## 📞 NEED HELP?

1. **Check:** COMPLETE_USER_GUIDE.md (comprehensive walkthrough)
2. **Check:** Browser console (F12) for errors
3. **Check:** Terminal output for server errors

**Contact:**  
Yang Guo & Marzieh Ghayour  
University of Alberta

---

## 🎯 EXPECTED RESULTS (If Hypothesis TRUE)

```
Correct choices:   M = 3.21s, SD = 1.42s
Incorrect choices: M = 5.47s, SD = 2.31s

t(487) = 8.34, p < 0.001

→ "Longer deliberation indicates uncertainty"
```

---

**You're ready to collect data! 🎉**

**Total setup time:** 5-10 minutes  
**Per session:** 20-40 minutes  
**Data collection:** Automatic (CSV logged in real-time)
