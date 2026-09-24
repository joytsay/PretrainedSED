<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import SoundIcon from '../components/SoundIcon.svelte';
  import '../app.css';
  import displayNamesText from '../../../../mid_to_display_name.tsv?raw';

  type WorkerEvent = {
    event: string;
    id?: number;
    cam_id?: string;
    timestamp_ms?: number;
    processing_ms?: number;
    superseded_packets?: number;
    signal_db?: number;
    signal_frequency_hz?: number;
    scores?: number[];
    classes?: string[];
    message?: string;
  };

  type PlaylistItem = { name: string; size: number; url: string; file?: File };
  type ScoreItem = { name: string; score: number; index: number };
  type HeapPerformance = Performance & { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } };

  const SAMPLE_RATE = 16000;
  const PACKET_MS = 40;
  const PACKET_SAMPLES = 640;
  const displayNames = displayNamesText
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => {
      const [mid, ...nameParts] = line.split('\t');
      return { mid, name: nameParts.join('\t') };
    });

  let video: HTMLVideoElement;
  let fileInput: HTMLInputElement;
  let mappingInput: HTMLInputElement;
  let playlist: PlaylistItem[] = [];
  let currentIndex = -1;
  let classes: string[] = [];
  let scores: number[] = [];
  let signalDb: number | null = null;
  let signalFrequencyHz: number | null = null;
  let thresholdPercent = 10;
  let camId = 'camera-01';
  let tabCameraId = '';
  let mappingText = '';
  let connected = false;
  let workerReady = false;
  let status = 'Connecting to callback worker…';
  let statusKind: '' | 'error' | 'warning' = '';
  let eventSequence = 0;
  let audioBuffer: AudioBuffer | null = null;
  let decoding = false;
  let streamId: number | null = null;
  let nextPacketTimestamp = 0;
  let requestCounter = 0;
  let packetSending = false;
  let pendingAutoPlay = false;
  let pumpTimer: number | undefined;
  let workerRecoveryTimer: number | undefined;
  let stopped = false;
  let seekWasPlaying = false;
  let suppressSeekRestart = false;
  let resumeAfterWorkerRestart = false;
  let realtimeContext: AudioContext | null = null;
  let realtimeSource: MediaElementAudioSourceNode | null = null;
  let realtimeProcessor: ScriptProcessorNode | null = null;
  let realtimeSamples: number[] = [];
  let realtimePacketChain = Promise.resolve();
  let realtimePendingPackets = 0;
  let realtimeBacklogRestarting = false;
  let workerWakeInFlight = false;
  let showSedTimeline = false;
  let timelineCanvas: HTMLCanvasElement;
  let visualizationTimer: number | undefined;
  let playlistAdvancing = false;
  let debugMode = false;
  let debugRunning = false;
  let debugTimer: number | undefined;
  let debugStartedAt = 0;
  let debugUptimeSeconds = 0;
  let debugCompletedMedia = 0;
  let debugMediaErrors = 0;
  let debugResults = 0;
  let debugAudioCallbacks = 0;
  let debugIdleAudioCallbacks = 0;
  let debugPacketsQueued = 0;
  let debugPacketsSent = 0;
  let debugPendingPackets = 0;
  let debugMaxPendingPackets = 0;
  let debugActiveRequests = 0;
  let debugRequestFailures = 0;
  let debugLastRequestMs = 0;
  let debugLastResultAt = 0;
  let debugLastResultAgeSeconds = 0;
  let debugLastInferenceMs = 0;
  let debugLastLagMs = 0;
  let debugLastSuperseded = 0;
  let debugPlaybackSeconds = 0;
  let debugDurationSeconds = 0;
  let debugHeapMb: number | null = null;
  let debugLastProgressAt = 0;
  let debugPreviousPosition = 0;
  let debugStallReported = false;
  let debugCallbackGapReported = false;
  let debugBacklogReported = false;
  let debugLastReportAt = 0;

  const VISUALIZATION_INTERVAL_MS = 100;
  const VISUALIZATION_SECONDS = 10;
  const DEBUG_REPORT_INTERVAL_MS = 10000;
  const MAX_REALTIME_PENDING_PACKETS = 100;
  const AUDIO_POST_TIMEOUT_MS = 8000;

  $: currentItem = currentIndex >= 0 ? playlist[currentIndex] : null;
  $: orderedScores = classes
    .map((name, index): ScoreItem => ({ name, index, score: scores[index] ?? 0 }))
    .sort((left, right) => right.score - left.score);
  $: triggeredScores = orderedScores.filter((item) => item.score * 100 > thresholdPercent);
  $: inactiveScores = orderedScores.filter((item) => item.score * 100 <= thresholdPercent);
  $: mappingClasses = aggregateNames(mappingText);

  const classPalette = [
    '#ff4057', // red
    '#ff8a2b', // orange
    '#ffd43b', // yellow
    '#35d277', // green
    '#20d5df', // cyan
    '#557cff', // blue
    '#d64cff'  // magenta
  ];

  function colorFor(label: string): string {
    const mappedIndex = mappingClasses.indexOf(label);
    if (mappedIndex >= 0) return classPalette[mappedIndex % classPalette.length];

    // Keep a stable fallback while the mapping is still loading.
    const fallbackIndex = Array.from(label).reduce(
      (total, character) => total + (character.codePointAt(0) ?? 0), 0
    );
    return classPalette[fallbackIndex % classPalette.length];
  }

  const categoryStyles = [
    { key: 'emergency sounds', aliases: ['siren', 'scream', 'alarm', 'ambulance', 'emergency'], title: 'Emergency Sounds', subtitle: 'Siren / Scream', color: '#ff4148', icon: 'siren' },
    { key: 'violence sounds', aliases: ['gunshot', 'gunfire', 'machine gun', 'battle cry', 'weapon', 'violence'], title: 'Violence Sounds', subtitle: 'Machine Gun / Gunshot / Battle Cry', color: '#8247ee', icon: 'violence' },
    { key: 'vehicle noise', aliases: ['engine', 'revving', 'race car', 'car horn', 'traffic', 'vehicle noise'], title: 'Vehicle Noise', subtitle: 'Modified Vehicle Noise / Engine Revving', color: '#5276f5', icon: 'vehicle' },
    { key: 'glass breaking', aliases: ['glass', 'shatter'], title: 'Glass Breaking', subtitle: 'Glass Breaking', color: '#79e88d', icon: 'glass' },
    { key: 'impact sounds', aliases: ['impact', 'collision', 'crash', 'smash', 'thump', 'thud', 'heavy object drop'], title: 'Impact Sounds', subtitle: 'Vehicle Collision / Heavy Object Drop', color: '#f47c20', icon: 'impact' },
    { key: 'explosion sounds', aliases: ['explosion', 'firecracker', 'firework', 'blast', 'detonation'], title: 'Explosion Sounds', subtitle: 'Explosion / Firecracker / Fireworks', color: '#ff3e43', icon: 'explosion' },
    { key: 'animal sounds', aliases: ['animal', 'dog', 'bark', 'canidae', 'wolf'], title: 'Animal Sounds', subtitle: 'Dog Bark', color: '#269957', icon: 'animal' }
  ];

  function categoryFor(name: string) {
    const normalized = name.trim().toLowerCase();
    const openParenthesis = name.indexOf('(');
    const title = (openParenthesis >= 0 ? name.slice(0, openParenthesis) : name).trim();
    const subtitle = openParenthesis >= 0
      ? name.slice(openParenthesis + 1).replace(/\)\s*$/, '').trim()
      : name;
    const key = title.toLowerCase();
    const exact = categoryStyles.find((item) => item.key === key);
    if (exact) return { ...exact, color: colorFor(name) };

    const similar = categoryStyles.find((item) =>
      item.aliases.some((alias) => normalized.includes(alias))
    );
    if (similar) return { ...similar, title, subtitle, color: colorFor(name) };

    return { title, subtitle, color: colorFor(name), icon: 'audio' };
  }

  function signalSummary(): string | null {
    if (signalDb === null || signalFrequencyHz === null) return null;
    const frequency = signalFrequencyHz >= 1000
      ? `${(signalFrequencyHz / 1000).toFixed(1)} kHz`
      : `${Math.round(signalFrequencyHz)} Hz`;
    return `${Math.round(signalDb)} dB · ${frequency}`;
  }

  function shiftCanvas(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement): number {
    const shift = Math.max(1, Math.round(
      canvas.width * VISUALIZATION_INTERVAL_MS / (VISUALIZATION_SECONDS * 1000)
    ));
    context.drawImage(canvas, shift, 0, canvas.width - shift, canvas.height, 0, 0, canvas.width - shift, canvas.height);
    context.clearRect(canvas.width - shift, 0, shift, canvas.height);
    return shift;
  }

  function resetSedTimeline(): void {
    if (timelineCanvas) {
      const context = timelineCanvas.getContext('2d');
      if (context) {
        context.fillStyle = '#06101e';
        context.fillRect(0, 0, timelineCanvas.width, timelineCanvas.height);
      }
    }
  }

  function drawSedTimeline(): void {
    if (!showSedTimeline || !timelineCanvas || !video || video.paused) return;

    const timelineContext = timelineCanvas.getContext('2d');
    if (!timelineContext) return;

    const shift = shiftCanvas(timelineContext, timelineCanvas);
    const rowHeight = timelineCanvas.height / Math.max(1, classes.length);
    for (let index = 0; index < classes.length; index += 1) {
      timelineContext.fillStyle = '#06101e';
      timelineContext.globalAlpha = 1;
      timelineContext.fillRect(timelineCanvas.width - shift, index * rowHeight, shift, Math.ceil(rowHeight));
      if ((scores[index] ?? 0) * 100 > thresholdPercent) {
        timelineContext.fillStyle = colorFor(classes[index]);
        timelineContext.fillRect(timelineCanvas.width - shift, index * rowHeight, shift, Math.ceil(rowHeight));
      }
    }
  }

  function toggleSedTimeline(event: Event): void {
    showSedTimeline = (event.currentTarget as HTMLInputElement).checked;
    if (showSedTimeline) requestAnimationFrame(resetSedTimeline);
  }

  function parseCsvRow(line: string): string[] {
    const fields: string[] = [];
    let field = '';
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      if (quoted && char === '"' && line[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = !quoted;
      } else if (char === ',' && !quoted) {
        fields.push(field.trim());
        field = '';
      } else {
        field += char;
      }
    }
    fields.push(field.trim());
    return fields;
  }

  function aggregateNames(csv: string): string[] {
    const seen = new Set<string>();
    const names: string[] = [];
    for (const line of csv.split(/\r?\n/)) {
      if (!line.trim() || line.trimStart().startsWith('#')) continue;
      const fields = parseCsvRow(line);
      if (fields[0]?.toLowerCase() === 'class_name') continue;
      if (fields[0] && !seen.has(fields[0])) {
        seen.add(fields[0]);
        names.push(fields[0]);
      }
    }
    return names;
  }

  async function api(path: string, options?: RequestInit): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      console.error('[SED] request failed', path, error);
      throw error;
    }
    if (!response.ok) {
      const body = await response.text();
      console.error('[SED] HTTP error', response.status, path, body);
      try { throw new Error(JSON.parse(body).error ?? body); }
      catch (error) {
        if (error instanceof SyntaxError) throw new Error(body || `HTTP ${response.status}`);
        throw error;
      }
    }
    return response;
  }

  async function loadMapping(): Promise<void> {
    try {
      mappingText = await (await api('/api/mapping')).text();
      status = `Loaded ${aggregateNames(mappingText).length} aggregate classes from class_mapping.csv`;
      statusKind = '';
    } catch (error) {
      showError(error);
    }
  }

  async function loadDefaultVideos(): Promise<void> {
    try {
      const payload = await (await api('/api/videos')).json();
      const defaults: PlaylistItem[] = (payload.videos ?? []).map(
        (item: { name: string; size: number; url: string }) => ({
          name: item.name,
          size: item.size,
          url: item.url
        })
      );
      if (defaults.length) {
        playlist = defaults;
        await selectItem(0);
        status = `Loaded ${defaults.length} media files from ${payload.root}`;
        statusKind = '';
      } else {
        status = `No media files found in ${payload.root}`;
        statusKind = 'warning';
      }
    } catch (error) {
      showError(error);
    }
  }

  async function saveMapping(): Promise<void> {
    try {
      await stopStream();
      workerReady = false;
      status = 'Validating mapping and restarting worker…';
      await api('/api/mapping', {
        method: 'PUT',
        headers: { 'Content-Type': 'text/csv; charset=utf-8' },
        body: mappingText
      });
    } catch (error) {
      showError(error);
    }
  }

  function downloadMapping(): void {
    const url = URL.createObjectURL(new Blob([mappingText], { type: 'text/csv' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'class_mapping.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function importMapping(event: Event): Promise<void> {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (file) mappingText = await file.text();
    input.value = '';
  }

  function addFiles(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    const additions = Array.from(input.files ?? []).map((file) => ({
      name: file.name,
      size: file.size,
      file,
      url: URL.createObjectURL(file)
    }));
    playlist = [...playlist, ...additions];
    if (currentIndex < 0 && playlist.length) selectItem(0);
    input.value = '';
  }

  async function clearPlaylist(): Promise<void> {
    video?.pause();
    await stopStream();
    for (const item of playlist) {
      if (item.file) URL.revokeObjectURL(item.url);
    }
    playlist = [];
    currentIndex = -1;
    audioBuffer = null;
    scores = classes.map(() => 0);
    signalDb = null;
    signalFrequencyHz = null;
    status = 'Playlist cleared';
  }

  async function selectItem(index: number, autoPlay = false): Promise<void> {
    if (index < 0 || index >= playlist.length) return;
    video?.pause();
    await stopStream();
    currentIndex = index;
    audioBuffer = null;
    resetSedTimeline();
    scores = classes.map(() => 0);
    signalDb = null;
    signalFrequencyHz = null;
    status = `Selected ${playlist[index].name}`;
    statusKind = '';
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    video.load();
    if (autoPlay) {
      // Start muted to satisfy browser autoplay policy, then restore sound once
      // playback has begun. The media source is still routed directly to the
      // speakers by setupRealtimeAudio().
      video.muted = true;
      try {
        await video.play();
        video.muted = false;
      } catch (error) {
        video.muted = false;
        showError(error);
      }
    }
  }

  async function decodeCurrent(): Promise<AudioBuffer> {
    if (audioBuffer) return audioBuffer;
    if (!currentItem) throw new Error('Choose at least one video or audio file');
    decoding = true;
    status = `Decoding audio from ${currentItem.name}…`;
    try {
      const AudioContextConstructor = window.AudioContext ?? window.webkitAudioContext;
      if (!AudioContextConstructor) throw new Error('This browser does not provide Web Audio');
      const context = new AudioContextConstructor();
      const encodedMedia = currentItem.file
        ? await currentItem.file.arrayBuffer()
        : await (await api(currentItem.url)).arrayBuffer();
      audioBuffer = await context.decodeAudioData(encodedMedia);
      await context.close();
      return audioBuffer;
    } finally {
      decoding = false;
    }
  }

  function pcmPacket(buffer: AudioBuffer, timestampMs: number): string {
    const bytes = new Uint8Array(PACKET_SAMPLES * 2);
    const view = new DataView(bytes.buffer);
    const sourceStart = timestampMs * buffer.sampleRate / 1000;
    for (let sample = 0; sample < PACKET_SAMPLES; sample += 1) {
      const sourcePosition = sourceStart + sample * buffer.sampleRate / SAMPLE_RATE;
      const left = Math.floor(sourcePosition);
      const fraction = sourcePosition - left;
      let value = 0;
      for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
        const data = buffer.getChannelData(channel);
        const a = left < data.length ? data[left] : 0;
        const b = left + 1 < data.length ? data[left + 1] : a;
        value += a + (b - a) * fraction;
      }
      value = Math.max(-1, Math.min(1, value / Math.max(1, buffer.numberOfChannels)));
      view.setInt16(sample * 2, value < 0 ? Math.round(value * 32768) : Math.round(value * 32767), true);
    }
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 1) binary += String.fromCharCode(bytes[offset]);
    return btoa(binary);
  }

  function pcmFloatPacket(samples: number[]): string {
    const bytes = new Uint8Array(PACKET_SAMPLES * 2);
    const view = new DataView(bytes.buffer);
    for (let index = 0; index < PACKET_SAMPLES; index += 1) {
      const value = Math.max(-1, Math.min(1, samples[index] ?? 0));
      view.setInt16(index * 2, value < 0 ? Math.round(value * 32768) : Math.round(value * 32767), true);
    }
    let binary = '';
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary);
  }

  async function sendMessage(message: Record<string, unknown>): Promise<void> {
    const startedAt = performance.now();
    const controller = message.type === 'audio' ? new AbortController() : null;
    const timeout = controller
      ? window.setTimeout(() => controller.abort(), AUDIO_POST_TIMEOUT_MS)
      : undefined;
    if (debugMode) debugActiveRequests += 1;
    try {
      await api('/api/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(message),
        signal: controller?.signal
      });
      if (debugMode && message.type === 'audio') debugPacketsSent += 1;
    } catch (error) {
      if (debugMode) debugRequestFailures += 1;
      throw error;
    } finally {
      if (timeout !== undefined) window.clearTimeout(timeout);
      if (debugMode) {
        debugActiveRequests -= 1;
        debugLastRequestMs = Math.round(performance.now() - startedAt);
      }
    }
  }

  function debugReport(): void {
    if (!debugMode || !debugRunning || !video) return;
    const now = performance.now();
    debugUptimeSeconds = Math.floor((now - debugStartedAt) / 1000);
    debugPlaybackSeconds = video.currentTime || 0;
    debugDurationSeconds = Number.isFinite(video.duration) ? video.duration : 0;
    debugLastResultAgeSeconds = debugLastResultAt ? Math.floor((now - debugLastResultAt) / 1000) : debugUptimeSeconds;
    const heap = (performance as HeapPerformance).memory;
    debugHeapMb = heap ? Math.round(heap.usedJSHeapSize / 1048576) : null;

    if (!video.paused && !video.ended && debugPlaybackSeconds > debugPreviousPosition + 0.1) {
      debugLastProgressAt = now;
      debugStallReported = false;
    }
    debugPreviousPosition = debugPlaybackSeconds;
    if (!video.paused && !video.ended && now - debugLastProgressAt > 15000 && !debugStallReported) {
      debugStallReported = true;
      console.timeStamp('SED playback stalled');
      console.warn('[SED debug] playback stalled', { media: currentItem?.name, positionSeconds: debugPlaybackSeconds, readyState: video.readyState, networkState: video.networkState });
    }
    if (!video.paused && debugLastResultAgeSeconds >= 15 && !debugCallbackGapReported) {
      debugCallbackGapReported = true;
      console.timeStamp('SED callback gap');
      console.warn('[SED debug] no inference callback for 15 seconds', { media: currentItem?.name, streamId, pendingPackets: debugPendingPackets, activeRequests: debugActiveRequests, workerReady, connected });
    }
    if (debugPendingPackets > 100 && !debugBacklogReported) {
      debugBacklogReported = true;
      console.warn('[SED debug] audio packet backlog', { pendingPackets: debugPendingPackets, maxPendingPackets: debugMaxPendingPackets });
    }
    if (debugPendingPackets < 20) debugBacklogReported = false;

    if (now - debugLastReportAt >= DEBUG_REPORT_INTERVAL_MS) {
      debugLastReportAt = now;
      console.info('[SED debug] stream snapshot', JSON.stringify({
        uptimeSeconds: debugUptimeSeconds,
        media: `${currentIndex + 1}/${playlist.length}`,
        name: currentItem?.name,
        completedMedia: debugCompletedMedia,
        mediaErrors: debugMediaErrors,
        positionSeconds: Number(debugPlaybackSeconds.toFixed(1)),
        durationSeconds: Number(debugDurationSeconds.toFixed(1)),
        paused: video.paused,
        streamId,
        workerReady,
        connected,
        results: debugResults,
        lastResultAgeSeconds: debugLastResultAgeSeconds,
        inferenceMs: debugLastInferenceMs,
        playbackLagMs: debugLastLagMs,
        superseded: debugLastSuperseded,
        audioCallbacks: debugAudioCallbacks,
        idleAudioCallbacks: debugIdleAudioCallbacks,
        packetsQueued: debugPacketsQueued,
        packetsSent: debugPacketsSent,
        pendingPackets: debugPendingPackets,
        maxPendingPackets: debugMaxPendingPackets,
        activeRequests: debugActiveRequests,
        requestFailures: debugRequestFailures,
        lastRequestMs: debugLastRequestMs,
        bufferedSamples: realtimeSamples.length,
        audioContextState: realtimeContext?.state ?? 'none',
        jsHeapMb: debugHeapMb,
        classes: classes.map((name, index) => ({ name, confidencePercent: Number(((scores[index] ?? 0) * 100).toFixed(1)), triggered: (scores[index] ?? 0) * 100 > thresholdPercent }))
      }));
    }
  }

  async function startDebugRun(): Promise<void> {
    if (!currentItem || currentItem.file || !workerReady || debugRunning) return;
    debugRunning = true;
    debugStartedAt = performance.now();
    debugLastProgressAt = debugStartedAt;
    debugPreviousPosition = video.currentTime || 0;
    debugCompletedMedia = 0;
    debugMediaErrors = 0;
    debugResults = 0;
    debugAudioCallbacks = 0;
    debugIdleAudioCallbacks = 0;
    debugPacketsQueued = 0;
    debugPacketsSent = 0;
    debugPendingPackets = 0;
    debugMaxPendingPackets = 0;
    debugRequestFailures = 0;
    debugLastResultAt = 0;
    debugCallbackGapReported = false;
    debugStallReported = false;
    debugBacklogReported = false;
    debugLastReportAt = 0;
    console.info('[SED debug] started', { media: `${currentIndex + 1}/${playlist.length}`, name: currentItem.name, thresholdPercent });
    console.timeStamp('SED debug run started');
    try {
      await startAnalysis(video.currentTime || 0, true);
    } catch (error) {
      debugRunning = false;
      showError(error);
      console.error('[SED debug] could not start', error);
    }
  }

  async function stopDebugRun(): Promise<void> {
    debugRunning = false;
    video.pause();
    await stopStream();
    console.info('[SED debug] stopped', { uptimeSeconds: debugUptimeSeconds, completedMedia: debugCompletedMedia, results: debugResults });
  }

  async function wakeWorkerIfStopped(): Promise<void> {
    if (workerWakeInFlight) return;
    workerWakeInFlight = true;
    try {
      const health = await (await api('/api/health')).json() as {
        worker_running?: boolean;
        worker_ready?: boolean;
        classes?: string[];
      };
      if (health.worker_ready) {
        if (health.classes?.length) {
          classes = health.classes;
          if (scores.length !== classes.length) scores = classes.map(() => 0);
        }
        workerReady = classes.length > 0;
        status = workerReady
          ? `TensorRT worker ready · ${classes.length} aggregate classes`
          : 'TensorRT worker ready; synchronizing classes…';
        statusKind = '';
        if (workerReady && resumeAfterWorkerRestart && video && currentItem) {
          resumeAfterWorkerRestart = false;
          void startAnalysis(video.currentTime, true).catch(showError);
        }
        return;
      }
      if (health.worker_running) {
        status = 'TensorRT worker is starting…';
        statusKind = 'warning';
        return;
      }

      workerReady = false;
      status = 'Waking TensorRT worker…';
      statusKind = 'warning';
      const id = newStreamId();
      const wakeCamId = 'watchdog';
      await sendMessage({
        type: 'stream_start', id, cam_id: wakeCamId, timestamp_ms: 0
      });
      // Close the synthetic stream after it has served its only purpose. Both
      // messages remain queued while TensorRT initializes, so this does not
      // leave an unused stream allocated in the worker.
      await sendMessage({
        type: 'stream_end', id, cam_id: wakeCamId, timestamp_ms: 0
      });
    } finally {
      workerWakeInFlight = false;
    }
  }

  function newStreamId(): number {
    requestCounter = (requestCounter + 1) % 1000;
    return Date.now() * 1000 + requestCounter;
  }

  async function registerTabCamera(): Promise<void> {
    try {
      const response = await api('/api/session');
      const payload = await response.json() as { camera_id?: string };
      tabCameraId = payload.camera_id ?? 'cam-01';
      camId = tabCameraId;
    } catch {
      tabCameraId = 'cam-01';
      camId = tabCameraId;
    }
  }

  function unregisterTabCamera(): void {
    // Camera IDs are server-assigned monotonic IDs and intentionally are not reused.
  }

  async function setupRealtimeAudio(): Promise<AudioContext> {
    const AudioContextConstructor = window.AudioContext ?? window.webkitAudioContext;
    if (!AudioContextConstructor) throw new Error('This browser does not provide Web Audio');
    if (!realtimeContext) {
      realtimeContext = new AudioContextConstructor({ sampleRate: SAMPLE_RATE });
      realtimeSource = realtimeContext.createMediaElementSource(video);
      realtimeProcessor = realtimeContext.createScriptProcessor(1024, 2, 1);
      realtimeSource.connect(realtimeProcessor);
      // Keep an independent, reliable audio path. Some WebAudio
      // implementations produce a silent ScriptProcessor output until its
      // output buffer is explicitly filled.
      realtimeSource.connect(realtimeContext.destination);
      // Keep normal video playback audible while copying its audio to packets.
      realtimeProcessor.connect(realtimeContext.destination);
      realtimeProcessor.onaudioprocess = (event) => {
        if (streamId === null || video.paused) {
          if (debugMode && debugRunning) debugIdleAudioCallbacks += 1;
          // Do not let Web Audio's callback clock run ahead of the media
          // element while playback is paused.
          realtimeSamples = [];
          return;
        }
        if (debugMode && debugRunning) debugAudioCallbacks += 1;
        const input = event.inputBuffer;
        const channels = input.numberOfChannels;
        const frames = input.length;
        const first = input.getChannelData(0);
        const second = channels > 1 ? input.getChannelData(1) : first;
        for (let index = 0; index < frames; index += 1) {
          const sample = channels > 1 ? (first[index] + second[index]) * 0.5 : first[index];
          realtimeSamples.push(sample);
        }
        const id = streamId;
        while (realtimeSamples.length >= PACKET_SAMPLES && streamId === id) {
          const packet = realtimeSamples.splice(0, PACKET_SAMPLES);
          const timestamp = nextPacketTimestamp;
          nextPacketTimestamp += PACKET_MS;
          realtimePendingPackets += 1;
          const debugTracked = debugMode && debugRunning;
          if (debugTracked) {
            debugPacketsQueued += 1;
            debugPendingPackets += 1;
            debugMaxPendingPackets = Math.max(debugMaxPendingPackets, debugPendingPackets);
          }
          realtimePacketChain = realtimePacketChain.then(async () => {
            try {
              if (streamId !== id || video.paused) return;
              await sendMessage({
                type: 'audio', id, cam_id: camId.trim(), timestamp_ms: timestamp,
                sample_rate: SAMPLE_RATE, channels: 1, encoding: 's16le',
                audio_b64: pcmFloatPacket(packet)
              });
            } finally {
              realtimePendingPackets -= 1;
              if (debugTracked) debugPendingPackets = Math.max(0, debugPendingPackets - 1);
            }
          }).catch((error) => {
            if (streamId !== id) return;
            showError(error);
            // Stop producing packets after a failed POST. Continuing to queue
            // one request every 40 ms can grow memory while the server is
            // unavailable. Health polling will reconnect and resume playback.
            resumeAfterWorkerRestart = true;
            workerReady = false;
            connected = false;
            video.pause();
            window.setTimeout(() => void stopStream(), 0);
          });
        }
        if (realtimePendingPackets >= MAX_REALTIME_PENDING_PACKETS && !realtimeBacklogRestarting) {
          realtimeBacklogRestarting = true;
          const position = video.currentTime;
          console.warn('[SED] audio POST backlog reached 100 packets; restarting stream at current playback position');
          video.pause();
          void startRealtimeAnalysis(position, true).catch((error) => {
            showError(error);
            resumeAfterWorkerRestart = true;
            workerReady = false;
          }).finally(() => { realtimeBacklogRestarting = false; });
        }
      };
    }
    await realtimeContext.resume();
    return realtimeContext;
  }

  async function startRealtimeAnalysis(positionSeconds: number, autoPlay: boolean): Promise<void> {
    video.pause();
    await stopStream();
    await setupRealtimeAudio();
    realtimeSamples = [];
    resetSedTimeline();
    realtimePacketChain = Promise.resolve();
    streamId = newStreamId();
    nextPacketTimestamp = Math.max(0, Math.floor(positionSeconds * 1000 / PACKET_MS) * PACKET_MS);
    pendingAutoPlay = false;
    const id = streamId;
    await sendMessage({ type: 'stream_start', id, cam_id: camId.trim(), timestamp_ms: nextPacketTimestamp });
    if (autoPlay) await video.play();
    status = `Streaming live 40 ms audio from ${currentItem?.name ?? 'media'}…`;
  }

  async function startAnalysis(positionSeconds = 0, autoPlay = true): Promise<void> {
    if (!workerReady) throw new Error('The TensorRT worker is not ready');
    if (!camId.trim()) throw new Error('Camera ID must not be empty');
    if (currentItem && !currentItem.file) {
      await startRealtimeAnalysis(positionSeconds, autoPlay);
      return;
    }
    video.pause();
    await stopStream();
    const buffer = await decodeCurrent();
    resetSedTimeline();
    const timestamp = Math.max(0, Math.floor(positionSeconds * 1000 / PACKET_MS) * PACKET_MS);
    streamId = newStreamId();
    nextPacketTimestamp = timestamp;
    pendingAutoPlay = autoPlay;
    const id = streamId;
    await sendMessage({ type: 'stream_start', id, cam_id: camId.trim(), timestamp_ms: timestamp });
    await sendNextPacket(buffer, id);
    status = `Pre-rolling ${currentItem?.name ?? 'media'} at ${timestamp} ms…`;
  }

  async function sendNextPacket(buffer: AudioBuffer, id: number): Promise<void> {
    const timestamp = nextPacketTimestamp;
    await sendMessage({
      type: 'audio', id, cam_id: camId.trim(), timestamp_ms: timestamp,
      sample_rate: SAMPLE_RATE, channels: 1, encoding: 's16le',
      audio_b64: pcmPacket(buffer, timestamp)
    });
    if (streamId === id) nextPacketTimestamp += PACKET_MS;
  }

  async function pumpPackets(): Promise<void> {
    if (packetSending || streamId === null || !audioBuffer || !video || video.paused || video.ended) return;
    packetSending = true;
    const id = streamId;
    try {
      const target = Math.floor(video.currentTime * 1000 / PACKET_MS) * PACKET_MS;
      while (streamId === id && nextPacketTimestamp <= target) await sendNextPacket(audioBuffer, id);
    } catch (error) {
      showError(error);
    } finally {
      packetSending = false;
    }
  }

  async function stopStream(): Promise<void> {
    const id = streamId;
    streamId = null;
    pendingAutoPlay = false;
    realtimeSamples = [];
    // Let the in-flight POST finish before stream_end. Queued packets see the
    // cleared stream id and drain without sending, so the queue stays bounded.
    await realtimePacketChain;
    if (id !== null) {
      try {
        await sendMessage({ type: 'stream_end', id, cam_id: camId.trim(), timestamp_ms: nextPacketTimestamp });
      } catch {
        // A mapping save may restart the worker before this close reaches it.
      }
    }
  }

  async function runCurrent(): Promise<void> {
    try { await startAnalysis(video.currentTime || 0, true); }
    catch (error) { showError(error); }
  }

  async function handleNativePlay(): Promise<void> {
    if (streamId !== null) {
      // Resume the existing stream. Its packet clock must remain contiguous;
      // seeking is handled separately by handleSeeked(), which starts a new
      // stream with a fresh timestamp.
      realtimeSamples = [];
      return;
    }
    if (decoding) return;
    try { await startAnalysis(video.currentTime, true); }
    catch (error) { showError(error); }
  }

  function handleSeeking(): void {
    seekWasPlaying = !video.paused;
  }

  async function handleSeeked(): Promise<void> {
    if (suppressSeekRestart) {
      suppressSeekRestart = false;
      return;
    }
    if (streamId === null) return;
    try { await startAnalysis(video.currentTime, seekWasPlaying); }
    catch (error) { showError(error); }
  }

  async function handleEnded(): Promise<void> {
    if (debugMode && debugRunning) {
      debugCompletedMedia += 1;
      console.info('[SED debug] media completed', { completedMedia: debugCompletedMedia, media: `${currentIndex + 1}/${playlist.length}`, name: currentItem?.name });
      console.timeStamp(`SED media completed ${debugCompletedMedia}`);
    }
    await advancePlaylist();
  }

  function mediaErrorDescription(): string {
    const error = video?.error;
    if (!error) return 'unknown media error';
    const descriptions: Record<number, string> = {
      [MediaError.MEDIA_ERR_ABORTED]: 'loading was aborted',
      [MediaError.MEDIA_ERR_NETWORK]: 'network loading failed',
      [MediaError.MEDIA_ERR_DECODE]: 'the browser could not decode the media',
      [MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED]: 'the URI or media format is unsupported'
    };
    return error.message || descriptions[error.code] || `media error ${error.code}`;
  }

  async function advancePlaylist(retryCurrent = false): Promise<void> {
    if (playlistAdvancing || !playlist.length) return;
    playlistAdvancing = true;
    const failures: string[] = [];
    const startIndex = currentIndex;
    try {
      await stopStream();
      for (let attempt = 0; attempt < playlist.length; attempt += 1) {
        const offset = attempt + (retryCurrent ? 0 : 1);
        const index = (startIndex + offset) % playlist.length;
        const item = playlist[index];
        try {
          await selectItem(index);
          await startAnalysis(0, true);
          return;
        } catch (error) {
          const reason = video?.error ? mediaErrorDescription() :
            (error instanceof Error ? error.message : String(error));
          console.error('[SED] skipping unplayable media', { name: item.name, url: item.url, reason });
          failures.push(`${item.name}: ${reason}`);
          await stopStream();
        }
      }
      status = `No playable media remains: ${failures.join('; ')}`;
      statusKind = 'error';
      if (debugMode && debugRunning) {
        debugRunning = false;
        console.error('[SED debug] run stopped: no playable media remains', failures);
      }
    } finally {
      playlistAdvancing = false;
    }
  }

  async function handleMediaError(): Promise<void> {
    if (!currentItem || playlistAdvancing) return;
    if (debugMode && debugRunning) debugMediaErrors += 1;
    const failed = currentItem;
    const reason = mediaErrorDescription();
    console.error('[SED] media playback failed', { name: failed.name, url: failed.url, reason });
    status = `Could not play ${failed.name}: ${reason}; reloading it…`;
    statusKind = 'warning';
    await advancePlaylist(true);
  }

  async function pollEvents(): Promise<void> {
    while (!stopped) {
      try {
        const payload = await (await api(`/api/events?after=${eventSequence}`)).json();
        connected = true;
        // The server periodically compacts its bounded event log and resets
        // its sequence.  Restart from zero so we receive the worker's ready
        // callback again instead of staying stuck at the old cursor.
        if (typeof payload.next === 'number' && payload.next < eventSequence) {
          eventSequence = 0;
          continue;
        }
        eventSequence = payload.next ?? eventSequence;
        for (const envelope of payload.events ?? []) handleWorkerEvent(envelope.data as WorkerEvent);
      } catch (error) {
        console.error('[SED] event polling failed', error);
        connected = false;
        if (!stopped) {
          status = `Server connection lost: ${error instanceof Error ? error.message : String(error)}`;
          statusKind = 'error';
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      }
    }
  }

  function handleWorkerEvent(event: WorkerEvent): void {
    if (event.event !== 'result') console.info('[SED] worker event', event);
    if (event.event === 'ready') {
      classes = event.classes ?? [];
      scores = classes.map(() => 0);
      workerReady = classes.length > 0;
      status = `TensorRT worker ready · ${classes.length} aggregate classes`;
      statusKind = '';
      if (resumeAfterWorkerRestart) {
        resumeAfterWorkerRestart = false;
        void startAnalysis(video.currentTime, true).catch(showError);
      }
      return;
    }
    if (event.event === 'worker_restarting') {
      resumeAfterWorkerRestart ||= Boolean(video && !video.paused);
      streamId = null;
      realtimeSamples = [];
      workerReady = false;
      status = 'Restarting worker with saved mapping…';
      return;
    }
    if (event.event === 'fatal' || event.event === 'server_error') {
      if (event.message?.includes('restarting')) {
        resumeAfterWorkerRestart ||= Boolean(video && !video.paused);
        streamId = null;
        realtimeSamples = [];
      }
      workerReady = false;
      status = event.message ?? 'Worker failed';
      statusKind = 'error';
      return;
    }
    if (event.event === 'worker_log') return;
    if (event.id !== streamId) return;
    if (event.event === 'error') {
      status = event.message ?? 'Analysis failed';
      statusKind = 'error';
    } else if (event.event === 'result' && event.scores?.length === classes.length) {
      if (debugMode && debugRunning) {
        debugResults += 1;
        debugLastResultAt = performance.now();
        debugLastInferenceMs = event.processing_ms ?? 0;
        debugLastSuperseded = event.superseded_packets ?? 0;
        debugCallbackGapReported = false;
      }
      scores = [...event.scores];
      signalDb = typeof event.signal_db === 'number' ? event.signal_db : null;
      signalFrequencyHz = typeof event.signal_frequency_hz === 'number'
        ? event.signal_frequency_hz
        : null;
      const timestamp = event.timestamp_ms ?? 0;
      const lag = Math.max(0, Math.round(video.currentTime * 1000) - timestamp);
      if (debugMode && debugRunning) debugLastLagMs = lag;
      const cameraLabel = event.id !== undefined ? `stream-${event.id}` : event.cam_id ?? 'stream';
      status = `${cameraLabel} · ${timestamp} ms · inference ${event.processing_ms ?? 0} ms · playback lag ${lag} ms · superseded ${event.superseded_packets ?? 0}`;
      statusKind = lag > 200 ? 'warning' : '';
      if (pendingAutoPlay) {
        pendingAutoPlay = false;
        suppressSeekRestart = true;
        video.currentTime = timestamp / 1000;
        video.play().catch(() => {
          status = 'Analysis is ready. Press play to continue (browser autoplay was blocked).';
          statusKind = 'warning';
        });
      }
    }
  }

  function showError(error: unknown): void {
    status = error instanceof Error ? error.message : String(error);
    statusKind = 'error';
    if (debugMode) console.error('[SED debug] frontend error', error);
  }

  function handleWindowError(event: ErrorEvent): void {
    if (debugMode) console.error('[SED debug] uncaught browser error', event.error ?? event.message);
  }

  function handleUnhandledRejection(event: PromiseRejectionEvent): void {
    if (debugMode) console.error('[SED debug] unhandled promise rejection', event.reason);
  }

  onMount(() => {
    debugMode = new URLSearchParams(window.location.search).get('debug') === '1';
    if (debugMode) {
      console.info('[SED debug] diagnostics enabled; start the run from the page');
      debugTimer = window.setInterval(debugReport, 1000);
      window.addEventListener('error', handleWindowError);
      window.addEventListener('unhandledrejection', handleUnhandledRejection);
    }
    void registerTabCamera().then(() => wakeWorkerIfStopped()).catch(showError);
    void loadMapping();
    void loadDefaultVideos();
    void pollEvents();
    pumpTimer = window.setInterval(() => void pumpPackets(), 20);
    workerRecoveryTimer = window.setInterval(() => {
      if (!workerReady && !workerWakeInFlight) void wakeWorkerIfStopped().catch(showError);
    }, 10000);
    visualizationTimer = window.setInterval(drawSedTimeline, VISUALIZATION_INTERVAL_MS);
  });

  onDestroy(() => {
    unregisterTabCamera();
    stopped = true;
    if (pumpTimer !== undefined) window.clearInterval(pumpTimer);
    if (workerRecoveryTimer !== undefined) window.clearInterval(workerRecoveryTimer);
    if (visualizationTimer !== undefined) window.clearInterval(visualizationTimer);
    if (debugTimer !== undefined) window.clearInterval(debugTimer);
    window.removeEventListener('error', handleWindowError);
    window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    for (const item of playlist) {
      if (item.file) URL.revokeObjectURL(item.url);
    }
    realtimeProcessor?.disconnect();
    realtimeSource?.disconnect();
    void realtimeContext?.close();
    void stopStream();
  });
</script>

<svelte:head><meta name="description" content="Browser testbed for the ATST-F TensorRT sound event detector" /></svelte:head>

<header class="topbar">
  <div class="brand"><img class="mark" src="/logo_cloud.svg" alt="" aria-hidden="true" /><div><strong>GeoVision SED</strong><span>Sound Event Detection</span></div></div>
  <button type="button" class="connection" class:online={connected && workerReady}
    class:actionable={connected && !workerReady} disabled={!connected || workerReady || workerWakeInFlight}
    title={connected && !workerReady ? 'Wake the TensorRT worker' : undefined}
    onclick={() => void wakeWorkerIfStopped().catch(showError)}>
    {workerWakeInFlight ? 'Waking worker…' : connected ? (workerReady ? 'Worker ready' : 'Worker stopped · Wake') : 'Offline'}
  </button>
</header>

<main>
  {#if debugMode}
    <section class="panel debug-panel" aria-label="Long running browser stream diagnostics">
      <div class="panel-head"><h2>Browser streaming debug run</h2><small>Device media · same Web UI pipeline</small></div>
      <div class="buttons">
        <button class="primary" onclick={() => void startDebugRun()} disabled={debugRunning || !workerReady || !currentItem || Boolean(currentItem?.file)}>Start continuous run</button>
        <button onclick={() => void stopDebugRun()} disabled={!debugRunning}>Stop</button>
        <span class="debug-state">{debugRunning ? 'Running until stopped' : 'Stopped'}</span>
      </div>
      <div class="debug-grid">
        <span>Uptime <b>{debugUptimeSeconds}s</b></span>
        <span>Media <b>{currentIndex + 1}/{playlist.length}</b></span>
        <span>Completed <b>{debugCompletedMedia}</b></span>
        <span>Media errors <b>{debugMediaErrors}</b></span>
        <span>Position <b>{debugPlaybackSeconds.toFixed(1)} / {debugDurationSeconds.toFixed(1)}s</b></span>
        <span>Results <b>{debugResults}</b></span>
        <span>Last callback <b>{debugLastResultAgeSeconds}s ago</b></span>
        <span>Inference <b>{debugLastInferenceMs} ms</b></span>
        <span>Playback lag <b>{debugLastLagMs} ms</b></span>
        <span>Superseded <b>{debugLastSuperseded}</b></span>
        <span>Queued / sent <b>{debugPacketsQueued} / {debugPacketsSent}</b></span>
        <span>Pending packets <b>{debugPendingPackets} (max {debugMaxPendingPackets})</b></span>
        <span>Active audio callbacks <b>{debugAudioCallbacks}</b></span>
        <span>Idle audio callbacks <b>{debugIdleAudioCallbacks}</b></span>
        <span>HTTP in flight <b>{debugActiveRequests}</b></span>
        <span>HTTP failures <b>{debugRequestFailures}</b></span>
        <span>Last POST <b>{debugLastRequestMs} ms</b></span>
        <span>JS heap <b>{debugHeapMb === null ? 'unavailable' : `${debugHeapMb} MB`}</b></span>
      </div>
      <progress class="debug-progress" value={debugPlaybackSeconds} max={debugDurationSeconds || 1} aria-label="Current media playback progress"></progress>
      <div class="debug-classes">
        {#each classes as name, index}
          <span class:triggered={(scores[index] ?? 0) * 100 > thresholdPercent}>
            <b>{name}</b> {(scores[index] ?? 0) * 100 > thresholdPercent ? 'TRIGGERED' : ''} {((scores[index] ?? 0) * 100).toFixed(1)}%
          </span>
        {/each}
      </div>
      <p class="debug-note">DevTools Console prints a snapshot every 10 seconds plus media changes, stalls, errors, and callback gaps. Keep this tab visible for a representative long run.</p>
    </section>
  {/if}
  <div class="studio">
    <div>
      <section class="panel preview-panel">
        <div class="panel-head"><h2>Media preview</h2><small>{currentItem?.name ?? 'No media selected'}</small></div>
        <div class="video-shell">
          <!-- svelte-ignore a11y_media_has_caption -->
          <video bind:this={video} src={currentItem?.url} controls playsinline
            onplay={() => void handleNativePlay()} onseeking={handleSeeking} onseeked={() => void handleSeeked()}
            onended={() => void handleEnded()} onerror={() => void handleMediaError()}></video>
          {#if !currentItem}<div class="empty-video">Choose one or more video/audio files</div>{/if}
        </div>
        <div class="controls-grid">
          <label>Camera ID<input name="camera-id" bind:value={camId} placeholder="camera-01" /></label>
          <label>Alert threshold <span>{thresholdPercent.toFixed(1)}%</span><input name="alert-threshold" type="range" min="0" max="100" step="0.5" bind:value={thresholdPercent} /></label>
          <label>Packet cadence<input name="packet-cadence" value="40 ms" disabled /></label>
          <label class="visualization-toggle">
            <input name="show-sed-timeline" type="checkbox" checked={showSedTimeline} onchange={toggleSedTimeline} />
            <span>Show SED timeline</span>
          </label>
        </div>
        {#if showSedTimeline}
          <section class="sed-visualization" aria-label="Live color-coded SED event timeline">
            <div class="visualization-head">
              <strong>Live SED events</strong>
              <small>Rolling {VISUALIZATION_SECONDS} seconds</small>
            </div>
            <div class="timeline-visualization">
              <div class="timeline-labels">
                {#each classes as name, index}
                  {@const category = categoryFor(name)}
                  {@const signal = signalSummary()}
                  <span style={`--label-color:${category.color}`} title={name}>
                    <b>{category.title}</b>
                    {#if (scores[index] ?? 0) * 100 > thresholdPercent && signal}
                      <small>{signal}</small>
                    {/if}
                  </span>
                {/each}
              </div>
              <canvas bind:this={timelineCanvas} width="720" height={Math.max(1, classes.length) * 24}
                style={`height:${Math.max(1, classes.length) * 24}px`}
                aria-label="Color-coded SED class confidence timeline"></canvas>
            </div>
          </section>
        {/if}
        <details class="media-files-panel">
          <summary>Media Files</summary>
          <div class="media-files-content">
            <div class="buttons">
              <input name="media-files" bind:this={fileInput} class="file-native" type="file" multiple accept="video/*,audio/*" onchange={addFiles} />
              <button class="primary" onclick={() => fileInput.click()}>Add media files</button>
              <button onclick={() => void runCurrent()} disabled={!currentItem || !workerReady || decoding}>Run current</button>
              <button class="danger" onclick={() => void clearPlaylist()} disabled={!playlist.length}>Clear</button>
            </div>
          </div>
        </details>
        {#if playlist.length}
          <ol class="playlist">
            {#each playlist as item, index}
              <li><button class:active={index === currentIndex} onclick={() => void selectItem(index, true)}><span>{item.name}</span><small>{(item.size / 1048576).toFixed(1)} MB</small></button></li>
            {/each}
          </ol>
        {/if}
        <div class:error={statusKind === 'error'} class:warning={statusKind === 'warning'} class="status">{status}</div>
      </section>
    </div>

    <div class="right-column">
      <section class="results-panel">
      <div class="panel-head"><h1>Sound Event Confidence</h1><small>{classes.length} live classes</small></div>
      <div class:active={triggeredScores.length > 0} class="triggered-events" aria-live="polite">
        <span class="triggered-heading">Triggered Sounds Events</span>
        {#if triggeredScores.length}
          {#each triggeredScores as item (item.index)}
            {@const category = categoryFor(item.name)}
            {@const signal = signalSummary()}
            <div class="triggered-card" style={`--label-color:${category.color}`}>
              <SoundIcon kind={category.icon} size={31} strokeWidth={1.9} />
              <div class="triggered-copy">
                <strong><b>{category.title}</b><b>{(item.score * 100).toFixed(1)}%</b></strong>
                {#if signal}<span class="triggered-signal">{signal}</span>{/if}
                <span>{category.subtitle}</span>
                <div class="bar triggered-confidence" style={`--score:${Math.min(100, item.score * 100)}%`}><i></i></div>
              </div>
            </div>
          {/each}
        {:else}
          <strong class="no-triggered-events">None</strong>
        {/if}
      </div>
      <div class="scores">
        {#each inactiveScores as item (item.index)}
          {@const category = categoryFor(item.name)}
          <div class="score-row"
            style={`--label-color:${category.color};--score:${Math.min(100, item.score * 100)}%`}>
            <div class="category-icon" aria-hidden="true"><SoundIcon kind={category.icon} size={34} strokeWidth={1.8} /></div>
            <div class="score-copy"><div class="score-line"><span>{category.title}</span><span>{(item.score * 100).toFixed(1)}%</span></div>
              <div class="subtitle">{category.subtitle}</div><div class="bar"><i></i></div></div>
          </div>
        {/each}
      </div>
      </section>

      <details class="panel mapping-panel">
        <summary class="mapping-summary"><strong>Class Mapping</strong><small>{mappingClasses.length} aggregate labels</small></summary>
        <div class="mapping-content">
          <p class="hero-note" style="max-width:none;text-align:left">Each row maps a model source class into an aggregate output class. Saving validates the CSV and restarts the worker.</p>
          <div class="badge-list">
            {#each mappingClasses as name}<span class="badge" style={`--label-color:${colorFor(name)}`}>{name}</span>{/each}
          </div>
          <textarea name="class-mapping" bind:value={mappingText} spellcheck="false" aria-label="Class mapping CSV"></textarea>
          <div class="buttons">
            <button class="primary" onclick={() => void saveMapping()}>Apply and save</button>
            <button onclick={() => void loadMapping()}>Reload saved</button>
            <input name="mapping-file" bind:this={mappingInput} class="file-native" type="file" accept=".csv,text/csv" onchange={(event) => void importMapping(event)} />
            <button onclick={() => mappingInput.click()}>Import CSV</button>
            <button onclick={downloadMapping}>Download CSV</button>
          </div>
          <details class="source-class-list">
            <summary>List all source classes <span>{displayNames.length}</span></summary>
            <div class="source-class-table">
              <div class="source-class-heading"><span>MID</span><span>Display name</span></div>
              {#each displayNames as item (item.mid)}
                <div class="source-class-entry"><code>{item.mid}</code><span>{item.name}</span></div>
              {/each}
            </div>
          </details>
        </div>
      </details>
    </div>
  </div>
</main>
