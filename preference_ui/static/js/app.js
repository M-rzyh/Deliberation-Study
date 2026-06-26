/**
 * PEBBLE Preference Collection UI - Main Application
 * Comprehensive timing tracking and user interaction handling
 */

// Application state
const APP_STATE = {
    currentComparison: null,
    condition: null,
    participantId: null,
    sessionStarted: false,
    choiceMade: null,
    confidenceScore: null,
    replayCountA: 0,
    replayCountB: 0,
    timerInterval: null,
    startTime: null,
    firstInteraction: false,
    promptShown: false,
    showTimerUI: true,
    sessionComplete: false
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 PEBBLE UI Initialized');

    const showTimerAttr = document.body?.dataset?.showTimerUi;
    APP_STATE.showTimerUI = (showTimerAttr !== 'false');
    
    // Get condition from page
    const conditionBadge = document.getElementById('condition-badge');
    if (conditionBadge) {
        APP_STATE.condition = conditionBadge.className.replace('condition-', '');
        APP_STATE.participantId = document.getElementById('participant-id').textContent.replace('Participant: ', '');
    }
    
    // Show condition-specific message
    showConditionMessage();
    
    // Set up event listeners
    setupEventListeners();
    
    // Enable debug mode with Ctrl+D
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'd') {
            e.preventDefault();
            toggleDebug();
        }
    });
});

function showConditionMessage() {
    const messageBox = document.getElementById('condition-message');
    const timingBox = document.getElementById('timing-display');
    let message = '';
    
    switch(APP_STATE.condition) {
        case 'baseline':
            message = 'Standard condition: Watch both trajectories carefully and choose which one is better.';
            break;
        case 'time_aware_opaque':
            message = 'Watch both trajectories carefully and choose which one is better.';
            break;
        case 'time_aware_transparent':
            message = '<strong>Note:</strong> Your response time will affect how your feedback is used in training. Faster responses on clear choices and slower responses on difficult choices are both valuable!';
            break;
        case 'explicit_confidence':
            message = 'After choosing, you will rate your confidence. Both certain and uncertain feedback is valuable!';
            break;
        case 'revision_enabled':
            message = 'You will be able to review and change your preferences at the end of the session.';
            break;
    }
    
    if (message) {
        messageBox.innerHTML = message;
        messageBox.style.display = 'block';
    }

    // Always show decision timer (trajectories shown -> A/B choice)
    if (timingBox) {
        timingBox.style.display = APP_STATE.showTimerUI ? 'block' : 'none';
    }
}

function setupEventListeners() {
    const startSessionBtn = document.getElementById('start-session-btn');
    if (startSessionBtn) {
        startSessionBtn.addEventListener('click', startSession);
    }

    const participantInput = document.getElementById('participant-name');
    const sessionInput = document.getElementById('session-id-input');
    if (participantInput && sessionInput) {
        participantInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                sessionInput.focus();
            }
        });
        sessionInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                startSession();
            }
        });
    }

    // Replay buttons
    document.getElementById('replay-a').addEventListener('click', () => replayTrajectory('A'));
    document.getElementById('replay-b').addEventListener('click', () => replayTrajectory('B'));
    
    // Choice buttons
    document.getElementById('choose-a').addEventListener('click', () => makeChoice('A'));
    document.getElementById('choose-b').addEventListener('click', () => makeChoice('B'));
    document.getElementById('choose-skip').addEventListener('click', () => makeChoice('SKIP'));
    
    // Confidence buttons
    const confidenceBtns = document.querySelectorAll('.confidence-btn');
    confidenceBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            submitConfidence(parseInt(this.dataset.value));
        });
    });
    
    // Next comparison button
    document.getElementById('next-comparison').addEventListener('click', onNextButtonClick);
    
    // Video event listeners for timing
    const videoA = document.getElementById('video-a');
    const videoB = document.getElementById('video-b');
    
    videoA.addEventListener('loadeddata', () => onVideoLoaded('A'));
    videoB.addEventListener('loadeddata', () => onVideoLoaded('B'));
    
    videoA.addEventListener('ended', () => onVideoEnded('A'));
    videoB.addEventListener('ended', () => onVideoEnded('B'));
    
    // Track first interaction (mouse movement or click)
    document.addEventListener('mousemove', markFirstInteraction, { once: true });
    document.addEventListener('click', markFirstInteraction, { once: true });
}

function openSessionSetupModal() {
    const modal = document.getElementById('session-setup-modal');
    if (modal) {
        modal.style.display = 'flex';
    }

    // Suggest next session id
    const sessionInput = document.getElementById('session-id-input');
    if (sessionInput && APP_STATE.currentComparison?.comparison_number) {
        const cur = parseInt(sessionInput.value || '1', 10);
        if (!Number.isNaN(cur)) {
            sessionInput.value = String(cur + 1);
        }
    }
}

function onNextButtonClick() {
    if (APP_STATE.sessionComplete) {
        openSessionSetupModal();
        return;
    }
    loadNextComparison();
}

async function startSession() {
    const participantInput = document.getElementById('participant-name');
    const sessionInput = document.getElementById('session-id-input');

    const participantId = (participantInput?.value || '').trim();
    const sessionId = (sessionInput?.value || '').trim();

    if (!participantId) {
        alert('Please enter participant ID (human name) first.');
        participantInput?.focus();
        return;
    }

    if (!sessionId) {
        alert('Please enter a session ID.');
        sessionInput?.focus();
        return;
    }

    try {
        const response = await fetch('/api/start_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                participant_id: participantId,
                session_id: sessionId
            })
        });

        const result = await response.json();
        if (!response.ok || result.status !== 'success') {
            throw new Error(result.message || 'Failed to start session');
        }

        if ((result.remaining_queries ?? 0) <= 0) {
            APP_STATE.sessionStarted = false;
            APP_STATE.sessionComplete = true;
            const sid = result.session_id ?? sessionId;
            alert(`No unlabeled queries remain for participant ${result.participant_id}, session ${sid}. Choose a different session ID and try again.`);
            sessionInput?.focus();
            sessionInput?.select?.();
            return;
        }

        APP_STATE.participantId = result.participant_id;
        if (typeof result.show_timer_ui !== 'undefined') {
            APP_STATE.showTimerUI = !!result.show_timer_ui;
            const timingBox = document.getElementById('timing-display');
            if (timingBox) {
                timingBox.style.display = APP_STATE.showTimerUI ? 'block' : 'none';
            }
        }
        APP_STATE.sessionStarted = true;
        APP_STATE.sessionComplete = false;

        const nextBtn = document.getElementById('next-comparison');
        nextBtn.textContent = 'Next Comparison';
        nextBtn.style.display = 'inline-block';

        document.getElementById('participant-id').textContent = `Participant: ${result.participant_id}`;
        document.getElementById('session-setup-modal').style.display = 'none';

        loadNextComparison();
    } catch (error) {
        console.error('Error starting session:', error);
        alert(`Could not start session: ${error.message}`);
    }
}

async function loadNextComparison() {
    if (!APP_STATE.sessionStarted) {
        return;
    }

    console.log('📥 Loading next comparison...');
    
    // Reset state
    APP_STATE.choiceMade = null;
    APP_STATE.confidenceScore = null;
    APP_STATE.replayCountA = 0;
    APP_STATE.replayCountB = 0;
    APP_STATE.firstInteraction = false;
    APP_STATE.promptShown = false;

    document.getElementById('elapsed-time').textContent = '0.0s';

    // Reset choice button state from previous comparison
    const chooseA = document.getElementById('choose-a');
    const chooseB = document.getElementById('choose-b');
    const chooseSkip = document.getElementById('choose-skip');
    [chooseA, chooseB, chooseSkip].forEach(btn => {
        btn.disabled = false;
        btn.style.background = '';
        btn.style.borderColor = '';
        btn.style.color = '';
    });

    // Reset confidence button state from previous comparison
    document.querySelectorAll('.confidence-btn').forEach(btn => {
        btn.disabled = false;
        btn.style.background = '';
        btn.style.borderColor = '';
        btn.style.color = '';
    });

    // Ensure next button is visible during normal flow
    document.getElementById('next-comparison').style.display = 'inline-block';
    
    // Show loading screen
    document.getElementById('loading-screen').style.display = 'flex';
    document.getElementById('comparison-area').style.display = 'none';
    document.getElementById('preference-section').style.display = 'none';
    document.getElementById('confidence-section').style.display = 'none';
    document.getElementById('feedback-message').style.display = 'none';
    
    // Reset timer
    if (APP_STATE.timerInterval) {
        clearInterval(APP_STATE.timerInterval);
    }
    APP_STATE.startTime = null;
    
    try {
        // Fetch comparison data
        const response = await fetch('/api/get_comparison');
        if (response.status === 410) {
            const done = await response.json();
            APP_STATE.sessionComplete = true;
            APP_STATE.sessionStarted = false;
            document.getElementById('loading-screen').style.display = 'none';
            document.getElementById('comparison-area').style.display = 'none';
            document.getElementById('preference-section').style.display = 'none';
            document.getElementById('confidence-section').style.display = 'none';

            const feedbackMsg = document.getElementById('feedback-message');
            const feedbackIcon = document.getElementById('feedback-icon');
            const feedbackText = document.getElementById('feedback-text');
            feedbackIcon.textContent = '✓';
            feedbackIcon.style.color = '#4caf50';
            feedbackText.innerHTML = `<strong>All done.</strong><br>${done.message || 'No more unlabeled queries.'}`;
            const nextBtn = document.getElementById('next-comparison');
            nextBtn.textContent = 'Start New Session';
            nextBtn.style.display = 'inline-block';
            feedbackMsg.style.display = 'block';
            return;
        }

        const data = await response.json();
        
        APP_STATE.currentComparison = data;
        console.log('✓ Loaded comparison:', data.comparison_id);
        
        // Update UI
        document.getElementById('count').textContent = data.comparison_number;
        
        // Hide loading, show comparison
        document.getElementById('loading-screen').style.display = 'none';
        document.getElementById('comparison-area').style.display = 'grid';

        // Load and play videos (visible to user)
        await loadTrajectories(data);
        
    } catch (error) {
        console.error('Error loading comparison:', error);
        alert('Error loading comparison. Please refresh the page.');
    }
}

async function loadTrajectories(data) {
    const videoA = document.getElementById('video-a');
    const videoB = document.getElementById('video-b');
    
    // Show loading overlays
    document.getElementById('loading-a').style.display = 'flex';
    document.getElementById('loading-b').style.display = 'flex';
    
    // Set video sources
    videoA.preload = 'auto';
    videoB.preload = 'auto';
    videoA.playsInline = true;
    videoB.playsInline = true;

    videoA.src = data.trajectory_a.video_path;
    videoB.src = data.trajectory_b.video_path;
    
    // Wait for both videos to load
    await Promise.all([
        new Promise(resolve => {
            videoA.addEventListener('loadeddata', resolve, { once: true });
        }),
        new Promise(resolve => {
            videoB.addEventListener('loadeddata', resolve, { once: true });
        })
    ]);

    // Mark timing: trajectories shown (both videos ready and visible)
    await markTiming('trajectories_shown');

    // Start timer on the first actual playback event (autoplay or manual play)
    const startTimerOnFirstPlay = () => {
        if (!APP_STATE.startTime) {
            startTimer();
        }
    };
    videoA.addEventListener('play', startTimerOnFirstPlay, { once: true });
    videoB.addEventListener('play', startTimerOnFirstPlay, { once: true });

    // Play videos sequentially: A then B
    try {
        videoA.currentTime = 0;
        await videoA.play();
        await new Promise(resolve => {
            videoA.addEventListener('ended', resolve, { once: true });
        });

        videoB.currentTime = 0;
        await videoB.play();
        await new Promise(resolve => {
            videoB.addEventListener('ended', resolve, { once: true });
        });

        showPreferencePrompt();
    } catch (error) {
        console.warn('Autoplay blocked, user must click play');
    }
}

function onVideoLoaded(trajectory) {
    console.log(`✓ Video ${trajectory} loaded`);
    
    // Hide loading overlay
    document.getElementById(`loading-${trajectory.toLowerCase()}`).style.display = 'none';
    
    // Show stats
    const video = document.getElementById(`video-${trajectory.toLowerCase()}`);
    const frames = Math.floor(video.duration * 30); // Assuming 30 FPS
    document.getElementById(`frames-${trajectory.toLowerCase()}`).textContent = frames;
    document.getElementById(`stats-${trajectory.toLowerCase()}`).style.display = 'block';
    
    // Enable replay button after video ends
    video.addEventListener('ended', function() {
        document.getElementById(`replay-${trajectory.toLowerCase()}`).disabled = false;
    }, { once: true });
}

function onVideoEnded(trajectory) {
    console.log(`Video ${trajectory} ended`);
}

function showPreferencePrompt() {
    if (APP_STATE.promptShown) {
        return;
    }
    APP_STATE.promptShown = true;

    console.log('Showing preference prompt');
    
    // Mark timing: prompt shown
    markTiming('prompt_shown');
    
    // Show preference section with animation
    const prefSection = document.getElementById('preference-section');
    prefSection.style.display = 'block';
    prefSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function replayTrajectory(trajectory) {
    console.log(`🔄 Replaying trajectory ${trajectory}`);
    
    const video = document.getElementById(`video-${trajectory.toLowerCase()}`);
    const replayBtn = document.getElementById(`replay-${trajectory.toLowerCase()}`);
    
    // Increment replay counter
    if (trajectory === 'A') {
        APP_STATE.replayCountA++;
        document.getElementById('replay-count-a').textContent = APP_STATE.replayCountA;
    } else {
        APP_STATE.replayCountB++;
        document.getElementById('replay-count-b').textContent = APP_STATE.replayCountB;
    }
    
    // Mark timing: replay start
    await markTiming(`replay_${trajectory.toLowerCase()}_start_${APP_STATE.replayCountA + APP_STATE.replayCountB}`);
    
    // Disable button during replay
    replayBtn.disabled = true;
    
    // Replay video
    video.currentTime = 0;
    await video.play();
    
    // Re-enable button when done
    video.addEventListener('ended', function() {
        replayBtn.disabled = false;
        markTiming(`replay_${trajectory.toLowerCase()}_end_${APP_STATE.replayCountA + APP_STATE.replayCountB}`);
    }, { once: true });
}

async function makeChoice(choice) {
    console.log(`✓ User chose: ${choice}`);
    
    APP_STATE.choiceMade = choice;

    // Stop decision timer at A/B selection
    if (APP_STATE.timerInterval) {
        clearInterval(APP_STATE.timerInterval);
    }
    
    // Mark timing: preference selected
    await markTiming('preference_selected');
    
    // Visual feedback
    const btnId = choice === 'SKIP' ? 'choose-skip' : `choose-${choice.toLowerCase()}`;
    const chosenBtn = document.getElementById(btnId);
    if (choice === 'SKIP') {
        chosenBtn.style.background = '#f59e0b';
        chosenBtn.style.borderColor = '#f59e0b';
        chosenBtn.style.color = 'white';
    } else {
        chosenBtn.style.background = '#4caf50';
        chosenBtn.style.borderColor = '#4caf50';
        chosenBtn.style.color = 'white';
    }
    
    // Disable both buttons
    document.getElementById('choose-a').disabled = true;
    document.getElementById('choose-b').disabled = true;
    document.getElementById('choose-skip').disabled = true;
    
    // Wait a moment for visual feedback
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Hide preference section
    document.getElementById('preference-section').style.display = 'none';
    
    // Check if we need to show confidence rating
    if (choice === 'SKIP') {
        await submitPreference(null, 'skipped');
    } else if (shouldShowConfidence()) {
        showConfidencePrompt();
    } else {
        // Submit without confidence
        await submitPreference(null, null);
    }
}

function shouldShowConfidence() {
    // Show confidence for explicit_confidence and baseline conditions
    return APP_STATE.condition === 'explicit_confidence' || APP_STATE.condition === 'baseline';
}

function showConfidencePrompt() {
    console.log('Showing confidence prompt');
    
    const confSection = document.getElementById('confidence-section');
    confSection.style.display = 'block';
    confSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function submitConfidence(score) {
    console.log(`✓ Confidence score: ${score}`);
    
    APP_STATE.confidenceScore = score;
    
    // Visual feedback
    document.querySelectorAll('.confidence-btn').forEach(btn => {
        btn.disabled = true;
        if (parseInt(btn.dataset.value) === score) {
            btn.style.background = '#4caf50';
            btn.style.borderColor = '#4caf50';
            btn.style.color = 'white';
        }
    });
    
    // Wait for visual feedback
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Submit preference
    await submitPreference(score, 'explicit');
}

async function submitPreference(confidence, confidenceMethod) {
    console.log('📤 Submitting preference...');
    
    // Mark timing: confidence submitted (final timing event)
    await markTiming('confidence_submitted');
    
    // Stop timer
    if (APP_STATE.timerInterval) {
        clearInterval(APP_STATE.timerInterval);
    }
    
    try {
        const response = await fetch('/api/submit_preference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                choice: APP_STATE.choiceMade,
                confidence: confidence,
                confidence_method: confidenceMethod
            })
        });
        
        const result = await response.json();
        console.log('✓ Submission successful:', result);
        
        // Hide confidence section
        document.getElementById('confidence-section').style.display = 'none';
        
        // Show feedback
        showFeedback(result);
        
    } catch (error) {
        console.error('Error submitting preference:', error);
        alert('Error submitting preference. Please try again.');
    }
}

function showFeedback(result) {
    const feedbackMsg = document.getElementById('feedback-message');
    const feedbackIcon = document.getElementById('feedback-icon');
    const feedbackText = document.getElementById('feedback-text');
    
    const skippedCount = Number(result.total_skipped || 0);
    const skippedBadge = `<span style="display:inline-block;margin-top:8px;padding:4px 10px;border-radius:999px;background:#fff3cd;color:#8a6d3b;border:1px solid #ffe69c;font-size:12px;font-weight:600;">Skipped: ${skippedCount}</span>`;

    // Customize feedback based on choice outcome
    if (APP_STATE.choiceMade === 'SKIP') {
        feedbackIcon.textContent = '-';
        feedbackIcon.style.color = '#f59e0b';
        feedbackText.innerHTML = `
            <strong>Skipped pair recorded.</strong><br>
            You have completed ${result.total_comparisons} comparison(s).<br>
            ${skippedBadge}<br>
            ${formatTimingSummary(result.timing_summary)}
        `;
    } else if (result.accuracy) {
        feedbackIcon.textContent = '✓';
        feedbackIcon.style.color = '#4caf50';
        feedbackText.innerHTML = `
            <strong>Correct!</strong><br>
            You have completed ${result.total_comparisons} comparison(s).<br>
            ${skippedBadge}<br>
            ${formatTimingSummary(result.timing_summary)}
        `;
    } else {
        feedbackIcon.textContent = '✗';
        feedbackIcon.style.color = '#f44336';
        feedbackText.innerHTML = `
            Comparison recorded.<br>
            You have completed ${result.total_comparisons} comparison(s).<br>
            ${skippedBadge}<br>
            ${formatTimingSummary(result.timing_summary)}
        `;
    }
    
    feedbackMsg.style.display = 'block';
    feedbackMsg.scrollIntoView({ behavior: 'smooth', block: 'center' });

    const nextBtn = document.getElementById('next-comparison');
    nextBtn.textContent = 'Next Comparison';
}

function formatTimingSummary(timing) {
    if (!timing) return '';
    
    let summary = '<div style="font-size: 14px; margin-top: 10px; color: #666;">';
    
    if (timing.total_feedback_time) {
        summary += `Total time: ${timing.total_feedback_time.toFixed(1)}s`;
    }

    if (timing.decision_time) {
        summary += ` | Decision: ${timing.decision_time.toFixed(1)}s`;
    }
    
    if (timing.deliberation_time) {
        summary += ` | Deliberation: ${timing.deliberation_time.toFixed(1)}s`;
    }
    
    if (timing.replay_count > 0) {
        summary += ` | Replays: ${timing.replay_count}`;
    }
    
    summary += '</div>';
    
    return summary;
}

async function markTiming(event) {
    try {
        await fetch('/api/mark_timing', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ event: event })
        });
        
        console.log(`⏱️ Timing marked: ${event}`);
        
    } catch (error) {
        console.error('Error marking timing:', error);
    }
}

function markFirstInteraction() {
    if (!APP_STATE.firstInteraction) {
        APP_STATE.firstInteraction = true;
        markTiming('first_interaction');
        console.log('👆 First interaction detected');
    }
}

function startTimer() {
    APP_STATE.startTime = Date.now();
    
    APP_STATE.timerInterval = setInterval(() => {
        const elapsed = (Date.now() - APP_STATE.startTime) / 1000;
        document.getElementById('elapsed-time').textContent = elapsed.toFixed(1) + 's';
    }, 100);
}

// Debug functions
function toggleDebug() {
    const debugPanel = document.getElementById('debug-panel');
    
    if (debugPanel.style.display === 'none') {
        debugPanel.style.display = 'block';
        updateDebugInfo();
        
        // Auto-update debug info
        setInterval(updateDebugInfo, 1000);
    } else {
        debugPanel.style.display = 'none';
    }
}

function updateDebugInfo() {
    const debugContent = document.getElementById('debug-content');
    const state = {
        'Comparison': APP_STATE.currentComparison?.comparison_id || 'None',
        'Condition': APP_STATE.condition,
        'Choice': APP_STATE.choiceMade || 'Not made',
        'Confidence': APP_STATE.confidenceScore || 'Not set',
        'Replays A': APP_STATE.replayCountA,
        'Replays B': APP_STATE.replayCountB,
        'First Interaction': APP_STATE.firstInteraction
    };
    
    let html = '<pre style="font-size: 11px;">';
    for (const [key, value] of Object.entries(state)) {
        html += `${key}: ${value}\n`;
    }
    html += '</pre>';
    
    debugContent.innerHTML = html;
}

async function showSessionSummary() {
    try {
        const response = await fetch('/api/session_summary');
        const summary = await response.json();
        
        const summaryContent = document.getElementById('summary-content');
        let html = '<div style="line-height: 1.8;">';
        
        for (const [key, value] of Object.entries(summary)) {
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            html += `<p><strong>${label}:</strong> ${value}</p>`;
        }
        
        html += '</div>';
        summaryContent.innerHTML = html;
        
        document.getElementById('summary-modal').style.display = 'flex';
        
    } catch (error) {
        console.error('Error loading summary:', error);
    }
}

function closeSummary() {
    document.getElementById('summary-modal').style.display = 'none';
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Only allow shortcuts if not typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    // A key - choose trajectory A
    if (e.key === 'a' || e.key === 'A') {
        if (APP_STATE.choiceMade === null && document.getElementById('preference-section').style.display !== 'none') {
            makeChoice('A');
        }
    }
    
    // B key - choose trajectory B
    if (e.key === 'b' || e.key === 'B') {
        if (APP_STATE.choiceMade === null && document.getElementById('preference-section').style.display !== 'none') {
            makeChoice('B');
        }
    }

    // '-' key - skip preference (label -1)
    if (e.key === '-') {
        if (APP_STATE.choiceMade === null && document.getElementById('preference-section').style.display !== 'none') {
            makeChoice('SKIP');
        }
    }
    
    // 1-5 keys - confidence
    if (e.key >= '1' && e.key <= '5') {
        if (document.getElementById('confidence-section').style.display !== 'none') {
            submitConfidence(parseInt(e.key));
        }
    }
    
    // Space or Enter - next comparison
    if (e.key === ' ' || e.key === 'Enter') {
        if (document.getElementById('feedback-message').style.display !== 'none') {
            e.preventDefault();
            onNextButtonClick();
        }
    }
    
    // S key - show summary
    if (e.key === 's' || e.key === 'S') {
        if (e.ctrlKey) {
            e.preventDefault();
            showSessionSummary();
        }
    }
});

console.log('📱 Keyboard shortcuts enabled:');
console.log('  A/B - Choose trajectory');
console.log('  -   - Skip (label -1)');
console.log('  1-5 - Confidence score');
console.log('  Space/Enter - Next comparison');
console.log('  Ctrl+S - Session summary');
console.log('  Ctrl+D - Debug panel');
