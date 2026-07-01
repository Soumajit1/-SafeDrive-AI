/* ════════════════════════════════════════════════════════════════════════
   Smart Cabin AI — dashboard.js
   Captures webcam frames in-browser, streams them to the Flask backend
   over WebSocket, and renders the returned telemetry + annotated frame.

   This page is the ONLY place that opens a long-lived getUserMedia()
   stream — permission.html only probes access and releases it again.
   ════════════════════════════════════════════════════════════════════════ */

(() => {
  const rawVideo       = document.getElementById('rawVideo');
  const captureCanvas  = document.getElementById('captureCanvas');
  const annotatedFrame = document.getElementById('annotatedFrame');
  const feedMsg        = document.getElementById('feedMsg');
  const recBadge       = document.getElementById('recBadge');

  const connDot   = document.getElementById('connDot');
  const connLabel = document.getElementById('connLabel');

  const alertBanner = document.getElementById('alertBanner');

  const gaugeArc    = document.getElementById('gaugeArc');
  const scoreVal    = document.getElementById('scoreVal');
  const gaugeStatus = document.getElementById('gaugeStatus');

  const mBlinks  = document.getElementById('mBlinks');
  const mEar     = document.getElementById('mEar');
  const mEarSub  = document.getElementById('mEarSub');
  const mClosed  = document.getElementById('mClosed');
  const mYaw     = document.getElementById('mYaw');
  const mYawSub  = document.getElementById('mYawSub');
  const mFace    = document.getElementById('mFace');
  const mFps     = document.getElementById('mFps');

  const logPanel = document.getElementById('logPanel');
  const earSpark = document.getElementById('earSpark');

  const resetBtn = document.getElementById('resetBtn');
  const stopBtn  = document.getElementById('stopBtn');
  const alarmAudio = document.getElementById('alarmAudio');

  const GAUGE_CIRCUMFERENCE = 603; // 2 * PI * r(96), matches stroke-dasharray
  const EAR_HIST_LEN = 60;
  let earHistory = new Array(EAR_HIST_LEN).fill(0.3);

  let ws = null;
  let stream = null;
  let lastAlertLevel = null;
  let lastDistractLog = 0;
  let sending = false;
  let stopped = false;
  let firstFrameReceived = false;
  let lastFrameSentAt = 0;
  let watchdogWarned = false;
  let alarmPlaying = false;

  // ── Logging helper ───────────────────────────────────────────────────
  function pushLog(message, level) {
    if (logPanel.querySelector('.log-empty')) logPanel.innerHTML = '';
    const row = document.createElement('div');
    row.className = 'log-row' + (level ? ' ' + level : '');
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    row.innerHTML = `<span>${message}</span><span class="ts">${ts}</span>`;
    logPanel.prepend(row);
    while (logPanel.children.length > 25) logPanel.removeChild(logPanel.lastChild);
  }

  // ── Feed status helper (always visible, never silently stuck) ─────────
  function setFeedMessage(html, showSpinner) {
    feedMsg.style.display = 'flex';
    feedMsg.innerHTML = (showSpinner
      ? '<span class="spinner" style="width:18px;height:18px;border-width:3px"></span>'
      : '') + `<span>${html}</span>`;
  }

  function showFeedError(html) {
    feedMsg.style.display = 'flex';
    feedMsg.innerHTML = `
      <span style="font-size:1.6rem">⚠️</span>
      <span style="max-width:320px;text-align:center;color:#ffb3b3">${html}</span>
      <button id="retryCamBtn" class="btn btn-ghost" style="margin-top:6px;padding:8px 18px;font-size:0.8rem">↺ Retry camera</button>
    `;
    const retryBtn = document.getElementById('retryCamBtn');
    if (retryBtn) retryBtn.addEventListener('click', () => {
      setFeedMessage('Retrying camera…', true);
      startCamera();
    });
  }

  // ── Alert banner ──────────────────────────────────────────────────────
  function renderAlert(data) {
    if (!data.face_detected) {
      alertBanner.className = 'alert-banner nodata';
      alertBanner.innerHTML = '👤 &nbsp;No driver face detected — please position yourself in front of the camera';
      return;
    }
    const map = {
      OK:      { cls: 'ok',      icon: '✅', text: 'Driver is alert — all systems normal' },
      FATIGUE: { cls: 'fatigue', icon: '⚠️', text: 'Fatigue detected — consider taking a break soon' },
      DROWSY:  { cls: 'drowsy',  icon: '🚨', text: 'DROWSINESS ALERT — pull over immediately' },
    };
    const m = map[data.alert_level] || map.OK;
    alertBanner.className = 'alert-banner ' + m.cls;
    alertBanner.innerHTML = `${m.icon} &nbsp;${m.text}`;
  }

  // ── Gauge ─────────────────────────────────────────────────────────────
  function renderGauge(score, level) {
    const offset = GAUGE_CIRCUMFERENCE * (1 - Math.min(100, Math.max(0, score)) / 100);
    gaugeArc.style.strokeDashoffset = offset.toFixed(1);
    const colors = { OK: '#3dd68c', FATIGUE: '#f5c842', DROWSY: '#ff4d4d' };
    gaugeArc.style.stroke = colors[level] || '#3dd68c';
    scoreVal.textContent = Math.round(score);
    scoreVal.style.color = colors[level] || '#eef2f7';

    const statusText = { OK: 'Alert', FATIGUE: 'Fatigue building', DROWSY: 'Drowsy — act now' };
    gaugeStatus.textContent = statusText[level] || '— Awaiting data —';
    gaugeStatus.style.color = colors[level] || '#93a1b5';
    gaugeStatus.style.borderColor = level ? (colors[level] + '55') : 'var(--line)';
  }

  // ── Emergency alarm — loops while drowsiness score is maxed out ───────
  function updateAlarm(score) {
    const shouldPlay = score >= 100;
    if (shouldPlay && !alarmPlaying) {
      alarmPlaying = true;
      alarmAudio.currentTime = 0;
      alarmAudio.play().catch(err => {
        // Browsers can block autoplay before any user gesture on the page.
        // The user has already clicked through permission.html by this
        // point, so this should be rare — but don't let it throw silently.
        console.warn('[Smart Cabin AI] alarm playback blocked:', err);
        alarmPlaying = false;
      });
      pushLog('🚨 Drowsiness score hit 100 — alarm triggered', 'crit');
    } else if (!shouldPlay && alarmPlaying) {
      alarmPlaying = false;
      alarmAudio.pause();
      alarmAudio.currentTime = 0;
    }
  }

  // ── Metrics ───────────────────────────────────────────────────────────
  function renderMetrics(data) {
    mBlinks.textContent = data.blink_count;

    mEar.textContent = data.avg_ear.toFixed(3);
    mEar.className = 'val ' + (data.avg_ear >= 0.22 ? 'c-green' : 'c-red');
    mEarSub.textContent = `L ${data.left_ear.toFixed(3)} · R ${data.right_ear.toFixed(3)}`;

    mClosed.textContent = data.eye_closed_frames;

    mYaw.textContent = `${data.head_yaw > 0 ? '+' : ''}${Math.round(data.head_yaw)}°`;
    mYaw.className = 'val ' + (data.distracted ? 'c-red' : 'c-green');
    mYawSub.textContent = data.distracted ? '⚠ distracted' : 'centred';

    mFace.textContent = data.face_detected ? '✓ Detected' : '— Not found';
    mFace.className = 'val ' + (data.face_detected ? 'c-green' : 'c-red');
    mFace.style.fontSize = '1rem';

    mFps.textContent = `${Math.round(data.fps)} fps`;
  }

  // ── EAR sparkline (SVG, no deps) ─────────────────────────────────────
  function renderSpark() {
    const W = 600, H = 60;
    const n = earHistory.length;
    const yOf = v => H - Math.min(1, Math.max(0, v / 0.5)) * H;
    const pts = earHistory.map((v, i) => `${(i * (W / (n - 1))).toFixed(1)},${yOf(v).toFixed(1)}`).join(' ');
    const thresholdY = yOf(0.22).toFixed(1);

    earSpark.innerHTML = `
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2dd4bf" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <line x1="0" y1="${thresholdY}" x2="${W}" y2="${thresholdY}" stroke="#ff4d4d" stroke-width="1" stroke-dasharray="4 3" opacity="0.45"/>
      <polygon points="0,${H} ${pts} ${W},${H}" fill="url(#sparkFill)"/>
      <polyline points="${pts}" fill="none" stroke="#2dd4bf" stroke-width="1.6"/>
    `;
  }

  // ── WebSocket setup ───────────────────────────────────────────────────
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/stream`);

    ws.onopen = () => {
      connDot.classList.add('live');
      connLabel.textContent = 'CONNECTED';
      pushLog('Connected to detection backend');
    };

    ws.onclose = () => {
      connDot.classList.remove('live');
      connLabel.textContent = stopped ? 'STOPPED' : 'DISCONNECTED';
      if (!stopped) pushLog('Backend connection lost — retrying…', 'crit');
      if (!stopped) setTimeout(connectWS, 2000);
    };

    ws.onerror = (e) => {
      connLabel.textContent = 'CONNECTION ERROR';
      console.error('[Smart Cabin AI] WebSocket error:', e);
    };

    ws.onmessage = (evt) => {
      let data;
      try { data = JSON.parse(evt.data); } catch { return; }
      if (data.reset) return;
      if (data.error) {
        pushLog(`Backend error: ${data.error}`, 'crit');
        console.error('[Smart Cabin AI] Backend reported error:', data.error);
        sending = false;
        return;
      }

      sending = false; // ready for next frame
      watchdogWarned = false;

      if (!firstFrameReceived) {
        firstFrameReceived = true;
        pushLog('First frame processed — detection running');
      }

      if (data.frame) {
        annotatedFrame.src = data.frame;
        annotatedFrame.style.display = 'block';
        feedMsg.style.display = 'none';
        recBadge.style.display = 'flex';
      }

      renderAlert(data);
      renderGauge(data.drowsiness_score, data.alert_level);
      renderMetrics(data);
      updateAlarm(data.drowsiness_score);

      earHistory.push(data.avg_ear);
      if (earHistory.length > EAR_HIST_LEN) earHistory.shift();
      renderSpark();

      // Event log
      if (data.alert_level !== lastAlertLevel) {
        const lvl = data.alert_level === 'DROWSY' ? 'crit' : (data.alert_level === 'FATIGUE' ? 'warn' : null);
        pushLog(`Status → ${data.alert_level}`, lvl);
        lastAlertLevel = data.alert_level;
      } else if (data.blink_detected) {
        pushLog(`Blink #${data.blink_count} detected (EAR ${data.avg_ear.toFixed(3)})`);
      } else if (data.distracted && (Date.now() - lastDistractLog > 3000)) {
        pushLog(`Distraction — head yaw ${data.head_yaw > 0 ? '+' : ''}${Math.round(data.head_yaw)}°`, 'warn');
        lastDistractLog = Date.now();
      }
    };
  }

  // ── Webcam capture loop ───────────────────────────────────────────────
  async function startCamera() {
    setFeedMessage('Requesting camera…', true);

    // Defensive: if a previous attempt left a stream open, stop it first.
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showFeedError('This browser does not support camera access. Use an up-to-date Chrome, Edge, or Firefox.');
      return;
    }

    // Hard timeout: getUserMedia() can hang indefinitely on some Windows
    // camera drivers if the device wasn't fully released by a previous
    // tab/page. Without this, the UI would sit on "Waiting for camera…"
    // forever with no feedback — exactly the bug being fixed here.
    let settled = false;
    const timeoutId = setTimeout(() => {
      if (settled) return;
      settled = true;
      showFeedError('Camera is taking too long to start. It may be in use by another app or browser tab. Close it and retry.');
    }, 8000);

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });

      if (settled) {
        // Timeout already fired and showed an error/retry UI — discard
        // this late-resolving stream to avoid double-init.
        stream.getTracks().forEach(t => t.stop());
        return;
      }
      settled = true;
      clearTimeout(timeoutId);

      rawVideo.srcObject = stream;
      rawVideo.muted = true;
      rawVideo.playsInline = true;

      try {
        await rawVideo.play();
      } catch (playErr) {
        // Autoplay can be rejected in rare cases; retry once after a
        // user-gesture-free microtask tick, otherwise surface the error.
        console.warn('[Smart Cabin AI] video.play() rejected, retrying:', playErr);
        await new Promise(r => setTimeout(r, 150));
        await rawVideo.play();
      }

      setFeedMessage('Camera ready — connecting to detector…', true);
      pushLog('Camera stream started');

      captureCanvas.width = 640;
      captureCanvas.height = 480;
      const ctx = captureCanvas.getContext('2d');

      function captureLoop() {
        if (stopped) return;
        if (ws && ws.readyState === WebSocket.OPEN && !sending && rawVideo.videoWidth > 0) {
          ctx.save();
          ctx.translate(captureCanvas.width, 0);
          ctx.scale(-1, 1); // mirror to match natural selfie view
          ctx.drawImage(rawVideo, 0, 0, captureCanvas.width, captureCanvas.height);
          ctx.restore();
          const b64 = captureCanvas.toDataURL('image/jpeg', 0.7);
          sending = true;
          lastFrameSentAt = Date.now();
          ws.send(JSON.stringify({ frame: b64 }));
        }
        requestAnimationFrame(captureLoop);
      }
      captureLoop();

    } catch (err) {
      if (settled && timeoutId) clearTimeout(timeoutId);
      settled = true;
      console.error('[Smart Cabin AI] getUserMedia failed:', err);

      let msg = `Could not access the camera (${err.name || err.message || 'unknown error'}).`;
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Camera access was denied. Click the camera icon in your browser\'s address bar, allow access, then retry.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No camera was found on this device.';
      } else if (err.name === 'NotReadableError') {
        msg = 'The camera is already in use by another app or browser tab. Close it, then retry.';
      }
      showFeedError(msg);
      pushLog(`Camera error: ${err.name || err.message}`, 'crit');
    }
  }

  // ── Controls ──────────────────────────────────────────────────────────
  resetBtn.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ cmd: 'reset' }));
      pushLog('Counters reset by user');
      earHistory = new Array(EAR_HIST_LEN).fill(0.3);
      renderSpark();
    }
  });

  stopBtn.addEventListener('click', () => {
    stopped = true;
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (ws) ws.close();
    alarmPlaying = false;
    alarmAudio.pause();
    alarmAudio.currentTime = 0;
    feedMsg.style.display = 'flex';
    feedMsg.innerHTML = '<span>■ Session stopped</span>';
    annotatedFrame.style.display = 'none';
    recBadge.style.display = 'none';
    alertBanner.className = 'alert-banner';
    alertBanner.innerHTML = '⏹️ &nbsp;Session stopped — refresh the page to start again';
    pushLog('Session stopped by user', 'warn');
    stopBtn.disabled = true;
    resetBtn.disabled = true;
  });

  // Release the camera if the user navigates away or closes the tab.
  window.addEventListener('beforeunload', () => {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (ws) ws.close();
    alarmAudio.pause();
  });

  // ── Watchdog: detect a stalled pipeline ──────────────────────────────
  // If a frame was sent but no response (and no error) has come back
  // within 5 seconds, something is wrong server-side (e.g. the backend
  // process isn't actually upgrading the WebSocket, or crashed mid-
  // request). Surface this instead of leaving the UI silently stuck.
  setInterval(() => {
    if (stopped || !sending || watchdogWarned) return;
    if (lastFrameSentAt && (Date.now() - lastFrameSentAt > 5000)) {
      watchdogWarned = true;
      pushLog('No response from backend in 5s — check the server terminal for errors', 'crit');
      connLabel.textContent = 'NOT RESPONDING';
      connDot.classList.remove('live');
    }
  }, 2000);

  // ── Boot ──────────────────────────────────────────────────────────────
  connectWS();
  startCamera();
})();