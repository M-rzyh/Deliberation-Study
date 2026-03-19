# Timing System Reference Guide

## Overview

The UI tracks **5 major timing metrics** to measure different aspects of human decision-making in preference feedback.

---

## Timing Metrics

### 1. **Total Feedback Time**
- **Definition:** Complete duration from when trajectories first appear to when confidence is submitted
- **Start event:** `trajectories_shown`
- **End event:** `confidence_submitted`
- **Calculation:** `confidence_submitted_time - trajectories_shown_time`
- **Use case:** Overall cognitive load measurement

**Example:**
```
Trajectories shown: 14:32:17.483
Confidence submitted: 14:32:50.201
Total feedback time: 32.718 seconds
```

---

### 2. **Initial Viewing Time**
- **Definition:** Time before participant makes any interaction
- **Start event:** `trajectories_shown`
- **End event:** `first_interaction`
- **Calculation:** `first_interaction_time - trajectories_shown_time`
- **Use case:** Passive observation period

**What counts as "first interaction":**
- Mouse movement
- Mouse click
- Keyboard press
- Clicking replay button

**Example:**
```
Trajectories shown: 14:32:17.483
First mouse move: 14:32:19.601
Initial viewing time: 2.118 seconds
```

---

### 3. **Deliberation Time** ⭐ PRIMARY METRIC
- **Definition:** Time from when preference prompt appears to when choice is made
- **Start event:** `prompt_shown`
- **End event:** `preference_selected`
- **Calculation:** `preference_selected_time - prompt_shown_time`
- **Use case:** **Core hypothesis testing** - hesitation as uncertainty indicator

**Example:**
```
Prompt shown: 14:32:30.100
Preference selected (A): 14:32:34.350
Deliberation time: 4.250 seconds
```

**Research hypothesis:**
- **Longer deliberation** → Higher uncertainty → More likely to be incorrect
- **Shorter deliberation** → Higher confidence → More likely to be correct

---

### 4. **Confidence Reporting Time**
- **Definition:** Time from making preference choice to submitting confidence score
- **Start event:** `preference_selected`
- **End event:** `confidence_submitted`
- **Calculation:** `confidence_submitted_time - preference_selected_time`
- **Use case:** Secondary decision speed, confidence calibration

**Example:**
```
Preference selected: 14:32:34.350
Confidence submitted (4): 14:32:36.180
Confidence reporting time: 1.830 seconds
```

---

### 5. **Replay Tracking**

#### 5a. **Replay Count**
- **Definition:** Total number of times participant replayed either trajectory
- **Tracked separately:** `replay_trajectory_a` and `replay_trajectory_b`
- **Total:** `replay_count = replay_trajectory_a + replay_trajectory_b`

#### 5b. **Total Replay Time**
- **Definition:** Cumulative time spent rewatching trajectories
- **Events:** `replay_a_start_N`, `replay_a_end_N`, `replay_b_start_N`, `replay_b_end_N`
- **Calculation:** Sum of all replay durations

**Example:**
```
Replay A (1st): 14:32:25.100 → 14:32:30.200 = 5.1s
Replay B (1st): 14:32:31.500 → 14:32:36.700 = 5.2s
Total replay time: 10.3 seconds
Replay count: 2
```

---

## Event Timeline (Example)

```
Time (s)    Event                           Metric Starts/Ends
─────────────────────────────────────────────────────────────
0.000       trajectories_shown              ← Total feedback time START
                                           ← Initial viewing time START

2.118       first_interaction              ← Initial viewing time END
            (mouse movement)                

15.617      prompt_shown                    ← Deliberation time START
            ("Which is better?")            

18.850      replay_a_start_1               ← Replay tracking

23.950      replay_a_end_1                 

27.867      preference_selected (A)         ← Deliberation time END
                                           ← Confidence reporting START

29.718      confidence_submitted (4)        ← Confidence reporting END
                                           ← Total feedback time END

RESULTS:
- Total feedback time: 29.718s
- Initial viewing: 2.118s
- Deliberation: 12.233s (27.867 - 15.617)
- Confidence reporting: 1.851s
- Replay time: 5.1s
- Replay count: 1
```

---

## Substitution Note: Deliberation vs. Rewatch

**Original Plan:** Use "rewatch time" as hesitation indicator

**Actual Implementation:** Use **deliberation time** instead

**Why?**
1. **More direct measure:** Deliberation = active decision-making period
2. **Less ambiguous:** Rewatch could be for various reasons (curiosity, confusion, etc.)
3. **Clearer hypothesis:** Longer deliberation directly indicates uncertainty
4. **Easier to measure:** Single clean interval vs. multiple replay intervals

**Both are tracked:**
- Deliberation time: Primary metric for uncertainty
- Replay behavior: Secondary metric for task difficulty / engagement

---

## Data Analysis Tips

### Correct Deliberation Time Calculation

```python
import pandas as pd
from datetime import datetime

df = pd.read_csv('preference_data/P01_session1_..._comparisons.csv')

# Already calculated in CSV
deliberation_times = df['deliberation_time']

# Or calculate from raw timestamps
df['prompt_time'] = pd.to_datetime(df['prompt_shown_time'])
df['choice_time'] = pd.to_datetime(df['preference_selected_time'])
df['deliberation_calculated'] = (df['choice_time'] - df['prompt_time']).dt.total_seconds()
```

### Primary Hypothesis Test

```python
import scipy.stats as stats

# Split by accuracy
correct = df[df['choice_accuracy'] == True]['deliberation_time']
incorrect = df[df['choice_accuracy'] == False]['deliberation_time']

# T-test
t_stat, p_value = stats.ttest_ind(correct, incorrect)

print(f"Mean deliberation (correct): {correct.mean():.2f}s")
print(f"Mean deliberation (incorrect): {incorrect.mean():.2f}s")
print(f"Difference: {incorrect.mean() - correct.mean():.2f}s")
print(f"p-value: {p_value:.4f}")

# Expected result if hypothesis is TRUE:
# Mean deliberation (correct): 3.21s
# Mean deliberation (incorrect): 5.47s
# Difference: 2.26s
# p-value: 0.0001
```

### Controlling for Difficulty

```python
# Difficulty confound: harder comparisons take longer AND are more error-prone
# Control by stratifying by difficulty

for difficulty in ['Easy', 'Medium', 'Hard']:
    subset = df[df['difficulty_category'] == difficulty]
    
    correct_subset = subset[subset['choice_accuracy'] == True]['deliberation_time']
    incorrect_subset = subset[subset['choice_accuracy'] == False]['deliberation_time']
    
    t_stat, p_val = stats.ttest_ind(correct_subset, incorrect_subset)
    
    print(f"\n{difficulty}:")
    print(f"  Correct: {correct_subset.mean():.2f}s (n={len(correct_subset)})")
    print(f"  Incorrect: {incorrect_subset.mean():.2f}s (n={len(incorrect_subset)})")
    print(f"  p-value: {p_val:.4f}")
```

---

## CSV Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `total_feedback_time` | float | Total time (trajectories → confidence) |
| `initial_viewing_time` | float | Time before first interaction |
| `deliberation_time` | float | **PRIMARY** - Prompt → choice |
| `confidence_reporting_time` | float | Choice → confidence |
| `total_replay_time` | float | Sum of all replays |
| `replay_count` | int | Total number of replays |
| `replay_trajectory_a` | int | Replays of trajectory A |
| `replay_trajectory_b` | int | Replays of trajectory B |
| `trajectories_shown_time` | datetime | Raw timestamp |
| `first_interaction_time` | datetime | Raw timestamp |
| `prompt_shown_time` | datetime | Raw timestamp |
| `preference_selected_time` | datetime | Raw timestamp |
| `confidence_submitted_time` | datetime | Raw timestamp |

---

## Common Issues

### Issue: `first_interaction_time` is None
**Cause:** User didn't move mouse or click before prompt appeared  
**Fix:** Auto-set to `prompt_shown_time` in analysis

### Issue: `deliberation_time` is very short (<0.5s)
**Cause:** User is rushing or button-mashing  
**Fix:** Flag as potential data quality issue; consider excluding

### Issue: `total_replay_time` doesn't match sum of individual replays
**Cause:** Overlapping replay periods or calculation error  
**Fix:** Use `replay_count` as primary metric, investigate timing logs

---

## Best Practices

1. **Always log timestamps first** - calculate durations in analysis
2. **Keep raw timestamps** - allows recalculation if needed
3. **Separate replay count from replay time** - different insights
4. **Use deliberation_time as primary** - most direct uncertainty measure
5. **Check for outliers** - times >30s may indicate distraction

---

## For Paper Writing

**Report deliberation time:**
> "Deliberation time was measured as the interval between presentation of the preference prompt and participant's selection, with a mean of 4.23s (SD=1.87) across all comparisons."

**Report hypothesis result:**
> "Incorrect choices were associated with significantly longer deliberation times (M=5.47s, SD=2.31) compared to correct choices (M=3.21s, SD=1.42), t(487)=8.34, p<0.001, Cohen's d=1.13."

**Report replay behavior:**
> "Participants replayed trajectories in 23.4% of comparisons (114/487), with higher replay rates in Hard difficulty comparisons (38.2%) compared to Easy (12.1%)."

---

## Questions?

Contact: Yang Guo & Marzieh Ghayour  
University of Alberta, Department of Computing Science
