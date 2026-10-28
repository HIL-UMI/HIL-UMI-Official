(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const colors = {
    grid: 'rgba(132, 164, 196, .16)', text: '#7890a8', cyan: '#36d9ff',
    red: '#ff5874', yellow: '#ffc857', model: '#55d7ff', human: '#ffc857'
  };
  const componentMeta = [
    ['position', 'XYZ', '#38d9ff'], ['rotation', 'ROTATION', '#8d7cff'],
    ['gripper', 'GRIPPER', '#ffc857']
  ];
  const actionLabels = ['X', 'Y', 'Z', 'R1·X', 'R1·Y', 'R1·Z', 'R2·X', 'R2·Y', 'R2·Z', 'Gripper'];
  const actionGroups = { position: [0, 1, 2], rotation: [3, 4, 5, 6, 7, 8], gripper: [9], all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] };

  let mode = 'live';
  let actionGroup = 'position';
  let actionEvent = null;
  let liveEvents = [];
  const liveByPrediction = new Map();
  let episodes = [];
  let replay = null;
  let replayEvent = null;
  let mainCamera = 'front';
  let latestAlignedOod = false;
  let liveCursor = 0;
  let livePollActive = false;
  let detectorMode = 'pi05';
  let detectorModeInitialized = false;
  const dashboardBuild = '20260905.2';

  function isAdvantage() { return detectorMode === 'advantage'; }
  function replayIsAdvantage() { return Boolean(replay && replay.manifest.detector_mode === 'advantage'); }

  function applyDetectorMode(nextMode) {
    if (nextMode !== 'pi05' && nextMode !== 'advantage') return;
    if (detectorModeInitialized && detectorMode === nextMode) return;
    detectorMode = nextMode;
    detectorModeInitialized = true;
    const advantage = isAdvantage();
    $('advantage-inputs').classList.toggle('hidden', !advantage);
    $('action-panel').classList.toggle('hidden', advantage);
    $('live-component-section').classList.toggle('hidden', advantage);
    $('identity-footer').innerHTML = advantage
      ? '<span>Frame identity</span> Reference RGB pair → current RGB pair → stage advantage → OOD decision → replay frames'
      : '<span>Frame identity</span> Model input RGB → prediction ID → aligned human window → Energy → replay frame';
    $('live-primary-label').textContent = advantage ? 'ADVANTAGE' : 'ENERGY';
    $('live-secondary-label').textContent = advantage ? 'THRESHOLD' : 'IMMEDIATE';
    $('live-frame-score-label').textContent = advantage ? 'CURRENT FRAME' : 'FRAME';
    $('live-aligned-label').textContent = advantage ? 'OFFSET' : 'ALIGNED MS';
    $('live-prefix-label').textContent = advantage ? 'REFERENCE FRAME' : 'PREFIX MS';
    $('live-history-title').textContent = advantage ? 'Historical Advantage Score curve' : 'Historical ES Score curve';
    $('live-history-legend').innerHTML = advantage
      ? '<i class="cyan"></i>Advantage <i class="red"></i>OOD sample <b>—</b> threshold'
      : '<i class="cyan"></i>Immediate <i class="red"></i>Aligned Energy <b>●</b> inference anchor';
    requestAnimationFrame(redrawAll);
  }

  function formatNumber(value, digits = 3) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
  }

  function formatTime(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    const rest = value - minutes * 60;
    return `${String(minutes).padStart(2, '0')}:${rest.toFixed(3).padStart(6, '0')}`;
  }

  function advantageFrameLabel(event, reference = false) {
    const index = Number(reference ? event.reference_frame_index : event.frame_index);
    if (Number.isFinite(index) && index >= 0) return `F${index}`;
    const ids = (reference ? event.reference_source_frame_ids : event.source_frame_ids) || {};
    return Number.isFinite(Number(ids.front)) ? `RGB${ids.front}` : 'LIVE';
  }

  function prepareCanvas(canvas, cssHeight) {
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(240, canvas.clientWidth || canvas.parentElement.clientWidth);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height: cssHeight };
  }

  function renderComponents(target, components) {
    target.innerHTML = '';
    componentMeta.forEach(([key, label, color]) => {
      const item = document.createElement('div');
      item.className = 'component-row';
      const score = components && components[key] ? Number(components[key].score) : NaN;
      const share = components && components[key] ? Number(components[key].share) : 0;
      item.innerHTML = `<div class="component-label"><span>${label}</span><b>${formatNumber(score)}</b><em>${Number.isFinite(share) ? Math.round(share * 100) : 0}%</em></div><div class="component-track"><i style="width:${Math.max(0, Math.min(100, share * 100))}%;background:${color}"></i></div>`;
      target.appendChild(item);
    });
  }

  function setMode(nextMode) {
    mode = nextMode;
    document.querySelectorAll('.mode-button').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
    document.querySelectorAll('.mode-view').forEach((view) => view.classList.toggle('active', view.id === `${mode}-view`));
    actionEvent = mode === 'replay' ? replayEvent : latestFinalizedLive();
    renderActions();
    requestAnimationFrame(redrawAll);
    if (mode === 'replay') loadEpisodes();
  }

  function latestFinalizedLive() {
    for (let index = liveEvents.length - 1; index >= 0; index -= 1) {
      if (isAdvantage() || liveEvents[index].finalized) return liveEvents[index];
    }
    return null;
  }

  function upsertLiveEvent(event) {
    const existing = liveByPrediction.get(event.prediction_id);
    // A delayed HTTP/WS copy of a provisional prediction must never regress a
    // prediction that has already received its finalized aligned Energy.
    if (!isAdvantage() && existing && existing.finalized && !event.finalized) return null;
    if (existing) Object.assign(existing, event);
    else {
      liveByPrediction.set(event.prediction_id, event);
      liveEvents.push(event);
    }
    if (liveEvents.length > 240) {
      const removed = liveEvents.shift();
      liveByPrediction.delete(removed.prediction_id);
    }
    return existing || event;
  }

  function renderMonitorStatus(status) {
    if (!status) return;
    const state = $('live-status');
    const paused = status.state !== 'running';
    const modeLabel = status.mode === 'recording' ? 'RECORDING' : 'IDLE MONITORING';
    $('live-camera-mode').textContent = paused ? String(status.state || 'starting').toUpperCase() : `${modeLabel} · 30 FPS`;
    $('live-camera-mode').classList.toggle('muted', paused);
    $('live-status-detail').textContent = status.detail || (paused ? status.reason : (status.mode === 'recording' ? 'Episode inference and replay capture are active' : 'Inference active; idle results are live-only'));
    state.classList.toggle('paused', paused);
    if (paused) {
      state.classList.remove('alert', 'normal');
      state.querySelector('small').textContent = 'INFERENCE';
      state.querySelector('strong').textContent = status.state === 'starting' ? 'STARTING' : 'HARDWARE PAUSED';
      $('live-anchor').textContent = String(status.reason || 'PAUSED').replaceAll('_', ' ').toUpperCase();
      $('live-anchor').classList.add('muted');
      ['live-energy', 'live-immediate', 'live-frame', 'live-model-latency', 'live-prediction-latency', 'live-aligned-latency', 'live-prefix-latency'].forEach((id) => { $(id).textContent = '—'; });
      if (!isAdvantage()) renderComponents($('live-components'), null);
      latestAlignedOod = false;
      actionEvent = null;
      if (mode === 'live') renderActions();
    }
  }

  function updateLive(event, deferCharts = false) {
    if (event.type === 'advantage_event') return updateAdvantageLive(event, deferCharts);
    event = upsertLiveEvent(event);
    if (!event) return;
    $('live-immediate').textContent = formatNumber(event.immediate_score);
    $('live-step').textContent = event.step;
    $('live-model-latency').textContent = formatNumber(event.model_latency_ms, 0);
    $('live-prediction-latency').textContent = formatNumber(event.prediction_latency_ms, 0);
    // The current provisional prediction is published after the preceding
    // aligned event. Keep the latest completed aligned latency visible.
    if (Number.isFinite(Number(event.aligned_latency_ms))) {
      $('live-aligned-latency').textContent = formatNumber(event.aligned_latency_ms, 0);
    }
    $('live-prefix-latency').textContent = formatNumber(
      event.model_breakdown_ms && event.model_breakdown_ms.prefix_ms,
      0,
    );
    const frameLabel = event.frame_index >= 0 ? `F${event.frame_index}` : 'LIVE';
    $('live-anchor').textContent = event.finalized ? `ALIGNED · ${frameLabel}` : `PREDICTING · ${frameLabel}`;
    $('live-anchor').classList.remove('muted');
    if (event.finalized) {
      $('live-frame').textContent = event.frame_index >= 0 ? event.frame_index : 'LIVE';
      $('live-energy').textContent = formatNumber(event.energy_score);
      renderComponents($('live-components'), event.energy_components);
      latestAlignedOod = Boolean(event.chunk_ood);
      actionEvent = event;
      if (mode === 'live' && !deferCharts) renderActions();
    }
    const ood = Boolean(latestAlignedOod || event.immediate_ood);
    const state = $('live-status');
    state.classList.remove('paused');
    state.classList.toggle('alert', ood);
    state.classList.toggle('normal', !ood);
    state.querySelector('small').textContent = event.capture_mode === 'recording' ? 'RECORDING DETECTION' : 'IDLE DETECTION';
    state.querySelector('strong').textContent = ood ? 'OOD DETECTED' : 'NORMAL';
    if (!deferCharts) drawLiveHistory();
  }

  function updateAdvantageLive(event, deferCharts = false) {
    event = upsertLiveEvent(event);
    if (!event) return;
    $('live-energy').textContent = formatNumber(event.advantage_score, 4);
    $('live-immediate').textContent = formatNumber(event.threshold, 3);
    $('live-frame').textContent = advantageFrameLabel(event);
    $('live-step').textContent = event.step;
    $('live-model-latency').textContent = formatNumber(event.model_latency_ms, 0);
    $('live-prediction-latency').textContent = formatNumber(event.prediction_latency_ms, 0);
    $('live-aligned-latency').textContent = `−${event.reference_offset}`;
    $('live-prefix-latency').textContent = advantageFrameLabel(event, true);
    $('live-anchor').textContent = `PAIR · ${advantageFrameLabel(event)} ↔ ${advantageFrameLabel(event, true)}`;
    $('live-anchor').classList.remove('muted');
    const state = $('live-status');
    state.classList.remove('paused');
    state.classList.toggle('alert', Boolean(event.ood));
    state.classList.toggle('normal', !event.ood);
    state.querySelector('small').textContent = event.capture_mode === 'recording' ? 'RECORDING ADVANTAGE' : 'IDLE ADVANTAGE';
    state.querySelector('strong').textContent = event.ood ? 'OOD DETECTED' : 'NORMAL';
    if (!deferCharts) drawLiveHistory();
  }

  function drawAxes(ctx, width, height, pad) {
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let line = 0; line <= 4; line += 1) {
      const y = pad.top + (line / 4) * (height - pad.top - pad.bottom);
      ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y);
    }
    ctx.stroke();
  }

  function drawLiveHistory() {
    const canvas = $('live-history');
    const { ctx, width, height } = prepareCanvas(canvas, 230);
    ctx.clearRect(0, 0, width, height);
    const events = liveEvents;
    if (!events.length) return drawEmpty(ctx, 'Waiting for inference anchors…', width, height);
    const pad = { left: 46, right: 18, top: 20, bottom: 28 };
    if (isAdvantage()) return drawAdvantageHistory(ctx, events, width, height, pad);
    const maxValue = Math.max(1, ...events.flatMap((event) => [Number(event.immediate_score) || 0, event.finalized ? Number(event.energy_score) || 0 : 0])) * 1.12;
    const x = (index) => pad.left + index / Math.max(1, events.length - 1) * (width - pad.left - pad.right);
    const y = (value) => height - pad.bottom - value / maxValue * (height - pad.top - pad.bottom);
    drawAxes(ctx, width, height, pad);
    drawSeries(ctx, events, 'immediate_score', colors.cyan, x, y, () => true);
    drawSeries(ctx, events, 'energy_score', colors.red, x, y, (event) => event.finalized);
    events.forEach((event, index) => {
      if (!event.finalized) return;
      ctx.fillStyle = colors.red; ctx.beginPath(); ctx.arc(x(index), y(event.energy_score), 3.2, 0, Math.PI * 2); ctx.fill();
    });
    drawScale(ctx, maxValue, height, pad);
  }

  function signedDomain(events, threshold) {
    const values = events.map((event) => Number(event.advantage_score)).filter(Number.isFinite).concat([Number(threshold) || 0]);
    let min = Math.min(...values); let max = Math.max(...values);
    const span = Math.max(.1, max - min);
    min -= span * .18; max += span * .18;
    if (min === max) { min -= .1; max += .1; }
    return { min, max };
  }

  function drawAdvantageHistory(ctx, events, width, height, pad) {
    const threshold = Number(events[events.length - 1].threshold) || 0;
    const domain = signedDomain(events, threshold);
    const x = (index) => pad.left + index / Math.max(1, events.length - 1) * (width - pad.left - pad.right);
    const y = (value) => pad.top + (domain.max - value) / (domain.max - domain.min) * (height - pad.top - pad.bottom);
    drawAxes(ctx, width, height, pad);
    ctx.strokeStyle = colors.yellow; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(pad.left, y(threshold)); ctx.lineTo(width - pad.right, y(threshold)); ctx.stroke(); ctx.setLineDash([]);
    drawSeries(ctx, events, 'advantage_score', colors.cyan, x, y, () => true);
    events.forEach((event, index) => {
      ctx.fillStyle = event.ood ? colors.red : colors.cyan;
      ctx.beginPath(); ctx.arc(x(index), y(Number(event.advantage_score)), event.ood ? 4 : 2.8, 0, Math.PI * 2); ctx.fill();
    });
    ctx.fillStyle = colors.text; ctx.font = '11px Inter, sans-serif';
    ctx.fillText(domain.max.toFixed(2), 4, pad.top + 4);
    ctx.fillText(domain.min.toFixed(2), 4, height - pad.bottom + 4);
    ctx.fillStyle = colors.yellow; ctx.fillText(`threshold ${threshold.toFixed(2)}`, pad.left + 5, y(threshold) - 6);
  }

  function drawSeries(ctx, events, key, color, x, y, include) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let drawing = false;
    events.forEach((event, index) => {
      if (!include(event)) { drawing = false; return; }
      if (!drawing) ctx.moveTo(x(index), y(Number(event[key]) || 0));
      else ctx.lineTo(x(index), y(Number(event[key]) || 0));
      drawing = true;
    });
    ctx.stroke();
  }

  function drawScale(ctx, maxValue, height, pad) {
    ctx.fillStyle = colors.text; ctx.font = '11px Inter, sans-serif';
    ctx.fillText(maxValue.toFixed(2), 5, pad.top + 4);
    ctx.fillText('0.00', 10, height - pad.bottom + 4);
  }

  function drawEmpty(ctx, text, width, height) {
    ctx.fillStyle = colors.text; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(text, width / 2, height / 2); ctx.textAlign = 'left';
  }

  function renderActions() {
    const grid = $('action-grid');
    grid.innerHTML = '';
    const dimensions = actionGroups[actionGroup];
    $('action-context').textContent = actionEvent && actionEvent.finalized
      ? `Prediction ${actionEvent.prediction_id} · video frame ${actionEvent.frame_index} · 15 human actions aligned to the model prefix`
      : 'Waiting for a finalized aligned window.';
    dimensions.forEach((dimension) => {
      const card = document.createElement('article');
      card.className = 'trajectory-card';
      card.innerHTML = `<div class="trajectory-title"><span>DIM ${String(dimension).padStart(2, '0')}</span><strong>${actionLabels[dimension]}</strong></div><canvas data-dimension="${dimension}"></canvas>`;
      grid.appendChild(card);
    });
    requestAnimationFrame(() => grid.querySelectorAll('canvas').forEach((canvas) => drawActionCanvas(canvas, Number(canvas.dataset.dimension), actionEvent)));
  }

  function drawActionCanvas(canvas, dimension, event) {
    const { ctx, width, height } = prepareCanvas(canvas, 190);
    ctx.clearRect(0, 0, width, height);
    const samples = event && event.model_samples_chunk;
    const human = event && event.human_chunk;
    if (!samples || !samples.length || !human || !human.length) return drawEmpty(ctx, 'No aligned trajectory', width, height);
    const values = samples.flatMap((sample) => sample.map((step) => Number(step[dimension]))).concat(human.map((step) => Number(step[dimension]))).filter(Number.isFinite);
    let min = Math.min(...values); let max = Math.max(...values);
    const span = Math.max(1e-6, max - min); min -= span * .12 || 1; max += span * .12 || 1;
    const pad = { left: 42, right: 12, top: 14, bottom: 27 };
    const x = (index) => pad.left + index / Math.max(1, human.length - 1) * (width - pad.left - pad.right);
    const y = (value) => pad.top + (max - value) / (max - min) * (height - pad.top - pad.bottom);
    drawAxes(ctx, width, height, pad);
    samples.forEach((sample) => drawPath(ctx, sample.map((step) => step[dimension]), x, y, 'rgba(85,215,255,.16)', 1));
    const means = human.map((_, index) => samples.reduce((sum, sample) => sum + Number(sample[index][dimension]), 0) / samples.length);
    const stds = means.map((mean, index) => Math.sqrt(samples.reduce((sum, sample) => sum + (Number(sample[index][dimension]) - mean) ** 2, 0) / samples.length));
    ctx.fillStyle = 'rgba(85,215,255,.13)'; ctx.beginPath();
    means.forEach((mean, index) => index ? ctx.lineTo(x(index), y(mean + stds[index])) : ctx.moveTo(x(index), y(mean + stds[index])));
    for (let index = means.length - 1; index >= 0; index -= 1) ctx.lineTo(x(index), y(means[index] - stds[index]));
    ctx.closePath(); ctx.fill();
    drawPath(ctx, means, x, y, colors.model, 2);
    drawPath(ctx, human.map((step) => step[dimension]), x, y, colors.human, 2.4);
    ctx.fillStyle = colors.text; ctx.font = '10px Inter, sans-serif';
    ctx.fillText(max.toFixed(3), 3, pad.top + 3); ctx.fillText(min.toFixed(3), 3, height - pad.bottom + 3);
    ctx.fillText('0', pad.left - 2, height - 8); ctx.fillText('14', width - pad.right - 13, height - 8);
  }

  function drawPath(ctx, values, x, y, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    values.forEach((value, index) => index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value)));
    ctx.stroke();
  }

  async function loadEpisodes() {
    try {
      const response = await fetch('/api/episodes', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      episodes = (await response.json()).episodes || [];
      renderEpisodeList();
    } catch (error) {
      $('episode-list').innerHTML = `<div class="empty-state">Replay catalog unavailable<br><small>${error.message}</small></div>`;
    }
  }

  function renderEpisodeList() {
    const list = $('episode-list'); list.innerHTML = '';
    if (!episodes.length) return list.innerHTML = '<div class="empty-state">No saved OOD replays yet.<br><small>Only writer-confirmed episodes are shown.</small></div>';
    episodes.forEach((episode) => {
      const button = document.createElement('button');
      button.className = `episode-item ${replay && replay.manifest.key === episode.key ? 'active' : ''}`;
      const date = new Date(Number(episode.started_at) * 1000);
      const advantage = episode.detector_mode === 'advantage';
      const mean = advantage ? episode.score_mean : episode.energy_mean;
      const maximum = advantage ? episode.score_max : episode.energy_max;
      button.innerHTML = `<span><b>EP ${String(Number(episode.episode_index) + 1).padStart(3, '0')}</b><time>${date.toLocaleString()}</time></span><small>${episode.run_id}</small><div><em>${advantage ? 'ADV' : 'PI05'}</em><em>${formatTime(episode.duration_s)}</em><em>μ ${formatNumber(mean, 2)}</em><em>max ${formatNumber(maximum, 2)}</em></div>`;
      button.addEventListener('click', () => selectEpisode(episode)); list.appendChild(button);
    });
  }

  async function selectEpisode(summary) {
    pauseReplay();
    const path = `/api/episodes/${encodeURIComponent(summary.run_id)}/${summary.episode_index}`;
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) return;
    replay = await response.json();
    replay.manifest.key = summary.key;
    replay.events.sort((a, b) => a.frame_index - b.frame_index);
    replayEvent = null; actionEvent = null;
    mainCamera = 'front';
    renderEpisodeList(); loadReplayVideos(); updateReplayAtTime(0); renderActions(); drawReplayEnergy();
    $('replay-title').textContent = `${summary.run_id} · Episode ${Number(summary.episode_index) + 1}`;
    $('replay-summary').textContent = `${summary.frame_count} frames · ${summary.event_count} aligned anchors · ${summary.codec}`;
    $('video-empty').classList.add('hidden');
    const advantage = replayIsAdvantage();
    $('advantage-replay-reference').classList.toggle('hidden', !advantage);
    $('replay-components').classList.toggle('hidden', advantage);
    $('replay-chart-title').textContent = advantage ? 'Stage Advantage' : 'Aligned Energy';
    $('replay-score-label').textContent = advantage ? 'HELD ADVANTAGE' : 'HELD ENERGY';
  }

  function videoUrl(camera) {
    const manifest = replay.manifest;
    return `/api/episodes/${encodeURIComponent(manifest.run_id)}/${manifest.episode_index}/video/${camera}`;
  }

  function loadReplayVideos(preserveTime = 0, shouldPlay = false) {
    if (!replay) return;
    const main = $('replay-main'); const secondary = $('replay-secondary');
    const secondaryCamera = mainCamera === 'front' ? 'wrist_left' : 'front';
    main.src = videoUrl(mainCamera); secondary.src = videoUrl(secondaryCamera);
    $('main-camera-label').textContent = mainCamera === 'front' ? 'FRONT' : 'WRIST LEFT';
    $('secondary-camera-label').textContent = secondaryCamera === 'front' ? 'FRONT' : 'WRIST LEFT';
    main.addEventListener('loadedmetadata', () => {
      main.currentTime = Math.min(preserveTime, main.duration || preserveTime);
      $('replay-scrubber').max = main.duration || replay.manifest.duration_s;
      if (shouldPlay) playReplay();
    }, { once: true });
    secondary.addEventListener('loadedmetadata', () => {
      secondary.currentTime = Math.min(preserveTime, secondary.duration || preserveTime);
    }, { once: true });
    main.load(); secondary.load();
    if (replayIsAdvantage()) {
      const referenceFront = $('replay-reference-front'); const referenceWrist = $('replay-reference-wrist');
      referenceFront.src = videoUrl('front'); referenceWrist.src = videoUrl('wrist_left');
      for (const video of [referenceFront, referenceWrist]) video.addEventListener('loadedmetadata', () => {
        if (replayEvent && Number.isFinite(Number(replayEvent.reference_frame_index))) {
          video.currentTime = Math.max(0, Number(replayEvent.reference_frame_index) / (Number(replay.manifest.fps) || 30));
        }
        video.pause();
      }, { once: true });
      referenceFront.load(); referenceWrist.load();
    }
  }

  function playReplay() {
    if (!replay) return;
    Promise.allSettled([$('replay-main').play(), $('replay-secondary').play()]);
    $('play-toggle').textContent = 'Ⅱ';
  }

  function pauseReplay() {
    $('replay-main').pause(); $('replay-secondary').pause(); $('play-toggle').textContent = '▶';
  }

  function heldEventAtFrame(frame) {
    if (!replay) return null;
    let held = null;
    for (const event of replay.events) {
      if (event.frame_index > frame) break;
      held = event;
    }
    return held;
  }

  function replayPhaseAtFrame(frame) {
    if (!replay) return 'RECORDING';
    const recordingStart = Number(replay.manifest.recording_start_frame);
    return Number.isFinite(recordingStart) && frame < recordingStart ? 'PRE-TRIGGER' : 'RECORDING';
  }

  function updateReplayAtTime(seconds) {
    if (!replay) return;
    const fps = Number(replay.manifest.fps) || 30;
    const frame = Math.max(0, Math.floor(seconds * fps + 1e-6));
    const replayPhase = replayPhaseAtFrame(frame);
    const event = heldEventAtFrame(frame);
    const actionChanged = (replayEvent && replayEvent.prediction_id) !== (event && event.prediction_id);
    replayEvent = event; actionEvent = event;
    $('replay-clock').textContent = `${formatTime(seconds)} / ${formatTime(replay.manifest.duration_s)}`;
    $('replay-scrubber').value = seconds;
    if (!event) {
      $('alignment-badge').textContent = `${replayPhase} · BEFORE FIRST ANCHOR`;
      $('replay-energy').textContent = '—'; $('replay-anchor-frame').textContent = 'No preceding inference anchor';
      renderComponents($('replay-components'), null);
      if (replayIsAdvantage()) $('replay-reference-label').textContent = 'NO ANCHOR';
    } else {
      const exact = frame === Number(event.frame_index);
      $('alignment-badge').textContent = exact
        ? `${replayPhase} · ANCHOR F${event.frame_index}`
        : `${replayPhase} · HELD FROM F${event.frame_index}`;
      $('alignment-badge').classList.toggle('exact', exact);
      if (replayIsAdvantage()) {
        $('replay-energy').textContent = formatNumber(event.advantage_score, 4);
        $('replay-anchor-frame').textContent = `Current F${event.frame_index} · reference F${event.reference_frame_index} · offset −${event.reference_offset} · model ${formatNumber(event.model_latency_ms, 0)} ms`;
        $('replay-ood-tag').textContent = event.ood ? 'OOD' : 'IN DISTRIBUTION';
        $('replay-ood-tag').classList.toggle('alert', Boolean(event.ood));
        $('replay-reference-label').textContent = `F${event.reference_frame_index}`;
        const referenceTime = Math.max(0, Number(event.reference_frame_index) / fps);
        for (const video of [$('replay-reference-front'), $('replay-reference-wrist')]) {
          if (video.readyState >= 1 && Math.abs((video.currentTime || 0) - referenceTime) > .02) video.currentTime = referenceTime;
          video.pause();
        }
      } else {
        $('replay-energy').textContent = formatNumber(event.energy_score);
        const alignedLatency = Number.isFinite(Number(event.aligned_latency_ms))
          ? Number(event.aligned_latency_ms)
          : (Number(event.evaluated_timestamp) - Number(event.anchor_timestamp)) * 1000;
        $('replay-anchor-frame').textContent = `Pi0.5 input frame ${event.frame_index} · model ${formatNumber(event.model_latency_ms, 0)} ms · prediction ${formatNumber(event.prediction_latency_ms, 0)} ms · aligned ${formatNumber(alignedLatency, 0)} ms`;
        $('replay-ood-tag').textContent = event.chunk_ood ? 'OOD' : 'IN DISTRIBUTION';
        $('replay-ood-tag').classList.toggle('alert', Boolean(event.chunk_ood));
        renderComponents($('replay-components'), event.energy_components);
      }
    }
    if (mode === 'replay' && actionChanged) renderActions();
    drawReplayEnergy(frame);
  }

  function drawReplayEnergy(cursorFrame = 0) {
    const canvas = $('replay-energy-chart');
    const { ctx, width, height } = prepareCanvas(canvas, 220);
    ctx.clearRect(0, 0, width, height);
    if (!replay || !replay.events.length) return drawEmpty(ctx, 'No finalized Energy anchors in this episode', width, height);
    const events = replay.events; const frameCount = Math.max(1, Number(replay.manifest.frame_count) - 1);
    const pad = { left: 45, right: 18, top: 18, bottom: 28 };
    if (replayIsAdvantage()) return drawAdvantageReplay(ctx, events, cursorFrame, frameCount, width, height, pad);
    const maxValue = Math.max(1, ...events.map((event) => Number(event.energy_score) || 0)) * 1.12;
    const x = (frame) => pad.left + frame / frameCount * (width - pad.left - pad.right);
    const y = (value) => height - pad.bottom - value / maxValue * (height - pad.top - pad.bottom);
    drawAxes(ctx, width, height, pad);
    ctx.strokeStyle = colors.red; ctx.lineWidth = 2; ctx.beginPath();
    events.forEach((event, index) => {
      const startX = x(event.frame_index); const nextFrame = index + 1 < events.length ? events[index + 1].frame_index : frameCount;
      const scoreY = y(event.energy_score);
      if (index === 0) ctx.moveTo(startX, scoreY); else ctx.lineTo(startX, scoreY);
      ctx.lineTo(x(nextFrame), scoreY);
    });
    ctx.stroke();
    events.forEach((event) => { ctx.fillStyle = colors.red; ctx.beginPath(); ctx.arc(x(event.frame_index), y(event.energy_score), 3.5, 0, Math.PI * 2); ctx.fill(); });
    ctx.strokeStyle = colors.yellow; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(cursorFrame), pad.top); ctx.lineTo(x(cursorFrame), height - pad.bottom); ctx.stroke();
    drawScale(ctx, maxValue, height, pad);
  }

  function drawAdvantageReplay(ctx, events, cursorFrame, frameCount, width, height, pad) {
    const threshold = Number(replay.manifest.score_threshold) || 0;
    const domain = signedDomain(events, threshold);
    const x = (frame) => pad.left + frame / frameCount * (width - pad.left - pad.right);
    const y = (value) => pad.top + (domain.max - value) / (domain.max - domain.min) * (height - pad.top - pad.bottom);
    drawAxes(ctx, width, height, pad);
    ctx.strokeStyle = colors.yellow; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(pad.left, y(threshold)); ctx.lineTo(width - pad.right, y(threshold)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = colors.cyan; ctx.lineWidth = 2; ctx.beginPath();
    events.forEach((event, index) => {
      const startX = x(event.frame_index); const nextFrame = index + 1 < events.length ? events[index + 1].frame_index : frameCount;
      const scoreY = y(event.advantage_score);
      if (index === 0) ctx.moveTo(startX, scoreY); else ctx.lineTo(startX, scoreY);
      ctx.lineTo(x(nextFrame), scoreY);
    });
    ctx.stroke();
    events.forEach((event) => { ctx.fillStyle = event.ood ? colors.red : colors.cyan; ctx.beginPath(); ctx.arc(x(event.frame_index), y(event.advantage_score), event.ood ? 4 : 3, 0, Math.PI * 2); ctx.fill(); });
    const recordingStart = Number(replay.manifest.recording_start_frame);
    if (Number.isFinite(recordingStart) && recordingStart > 0 && recordingStart <= frameCount) {
      const markerX = x(recordingStart);
      ctx.strokeStyle = colors.text; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(markerX, pad.top); ctx.lineTo(markerX, height - pad.bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = colors.text; ctx.font = '11px Inter, sans-serif'; ctx.fillText('A · RECORDING', Math.min(markerX + 5, width - 92), pad.top + 12);
    }
    ctx.strokeStyle = colors.yellow; ctx.beginPath(); ctx.moveTo(x(cursorFrame), pad.top); ctx.lineTo(x(cursorFrame), height - pad.bottom); ctx.stroke();
    ctx.fillStyle = colors.text; ctx.font = '11px Inter, sans-serif'; ctx.fillText(domain.max.toFixed(2), 4, pad.top + 4); ctx.fillText(domain.min.toFixed(2), 4, height - pad.bottom + 4);
  }

  function redrawAll() {
    drawLiveHistory();
    drawReplayEnergy(replay && $('replay-main').currentTime ? Math.floor($('replay-main').currentTime * replay.manifest.fps) : 0);
    renderActions();
  }

  function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/ws`);
    socket.onopen = () => {
      $('connection').textContent = `Video connected · ${dashboardBuild} · checking metrics`;
      $('connection-dot').className = 'online';
    };
    socket.onmessage = (message) => {
      let event;
      try {
        event = JSON.parse(message.data);
      } catch (error) {
        $('connection').textContent = 'Invalid server message';
        $('connection-dot').className = 'offline';
        console.error('Invalid dashboard JSON', error);
        return;
      }
      try {
        if (event.type === 'preview' && event.image_b64 && event.wrist_image_b64) {
          $('live-preview').src = `data:image/jpeg;base64,${event.image_b64}`;
          $('live-wrist-preview').src = `data:image/jpeg;base64,${event.wrist_image_b64}`;
          const frameIds = event.source_frame_ids || {};
          const frameIdentity = Number.isFinite(Number(frameIds.front)) && Number.isFinite(Number(frameIds.wrist_left))
            ? ` · F${frameIds.front} / W${frameIds.wrist_left}`
            : '';
          $('live-frame-label').textContent = `RGB PAIR ${new Date(event.timestamp * 1000).toLocaleTimeString()}${frameIdentity}`;
        } else if (event.type === 'advantage_anchor') {
          $('adv-current-front').src = `data:image/jpeg;base64,${event.current_front_b64}`;
          $('adv-current-wrist').src = `data:image/jpeg;base64,${event.current_wrist_b64}`;
          $('adv-reference-front').src = `data:image/jpeg;base64,${event.reference_front_b64}`;
          $('adv-reference-wrist').src = `data:image/jpeg;base64,${event.reference_wrist_b64}`;
          $('adv-current-label').textContent = advantageFrameLabel(event);
          $('adv-reference-label').textContent = advantageFrameLabel(event, true);
        } else if (event.type === 'prediction' || event.type === 'aligned_event' || event.type === 'event' || event.type === 'advantage_event') updateLive(event);
        else if (event.type === 'episode_ready') loadEpisodes();
      } catch (error) {
        $('connection').textContent = `Metric render error · ${dashboardBuild}`;
        $('connection-dot').className = 'offline';
        console.error('Failed to render dashboard event', event, error);
      }
    };
    socket.onclose = () => { $('connection').textContent = 'Reconnecting'; $('connection-dot').className = 'offline'; setTimeout(connectWebSocket, 1200); };
    socket.onerror = () => socket.close();
  }

  async function pollLiveMetrics() {
    if (livePollActive) return;
    livePollActive = true;
    try {
      const response = await fetch(`/api/live?after=${liveCursor}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      const serverBuild = String(payload.dashboard_build || 'unknown');
      if (serverBuild !== dashboardBuild) {
        throw new Error(`UI ${dashboardBuild} / server ${serverBuild}`);
      }
      applyDetectorMode(String(payload.detector_mode || 'pi05'));
      const metricEvents = (payload.events || []).filter((event) => (
        event.type === 'prediction' || event.type === 'aligned_event' || event.type === 'event' || event.type === 'advantage_event'
      ));
      metricEvents.forEach((event) => {
        updateLive(event, true);
      });
      if (metricEvents.length) {
        if (mode === 'live' && !isAdvantage()) renderActions();
        drawLiveHistory();
      }
      renderMonitorStatus(payload.monitor_status);
      liveCursor = Math.max(liveCursor, Number(payload.latest_sequence) || 0);
      $('connection').textContent = `Connected · ${dashboardBuild} · metrics ${liveCursor}`;
      $('connection-dot').className = 'online';
    } catch (error) {
      $('connection').textContent = `Metrics unavailable · ${error.message} · restart monitor`;
      $('connection-dot').className = 'offline';
      console.error('Live metric polling failed', error);
    } finally {
      livePollActive = false;
    }
  }

  document.querySelectorAll('.mode-button').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
  document.querySelectorAll('.action-tab').forEach((button) => button.addEventListener('click', () => {
    actionGroup = button.dataset.group;
    document.querySelectorAll('.action-tab').forEach((tab) => tab.classList.toggle('active', tab === button));
    renderActions();
  }));
  $('refresh-episodes').addEventListener('click', loadEpisodes);
  $('play-toggle').addEventListener('click', () => $('replay-main').paused ? playReplay() : pauseReplay());
  $('swap-camera').addEventListener('click', () => {
    if (!replay) return;
    const time = $('replay-main').currentTime; const playing = !$('replay-main').paused;
    pauseReplay(); mainCamera = mainCamera === 'front' ? 'wrist_left' : 'front'; loadReplayVideos(time, playing);
  });
  $('replay-main').addEventListener('timeupdate', () => {
    const main = $('replay-main'); const secondary = $('replay-secondary');
    if (Math.abs((secondary.currentTime || 0) - main.currentTime) > .06) secondary.currentTime = main.currentTime;
    updateReplayAtTime(main.currentTime);
  });
  $('replay-main').addEventListener('ended', pauseReplay);
  $('replay-scrubber').addEventListener('input', (event) => {
    const time = Number(event.target.value); $('replay-main').currentTime = time; $('replay-secondary').currentTime = time; updateReplayAtTime(time);
  });
  $('replay-energy-chart').addEventListener('click', (event) => {
    if (!replay) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - 45) / Math.max(1, rect.width - 63)));
    const time = fraction * replay.manifest.duration_s;
    $('replay-main').currentTime = time; $('replay-secondary').currentTime = time; updateReplayAtTime(time);
  });
  window.addEventListener('resize', redrawAll);

  renderComponents($('live-components'), null);
  renderComponents($('replay-components'), null);
  applyDetectorMode('pi05');
  renderActions(); drawLiveHistory(); drawReplayEnergy(); loadEpisodes(); connectWebSocket();
  pollLiveMetrics();
  window.setInterval(pollLiveMetrics, 500);
})();
