<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { AudioLines, Bomb, CarFront, Crosshair, Dog, GlassWater, Hammer, Siren } from '@lucide/svelte';
  import '../app.css';
  import displayNamesText from '../../../../mid_to_display_name.tsv?raw';

  type WorkerEvent = {
    event: string;
    id?: number;
    cam_id?: string;
    timestamp_ms?: number;
    processing_ms?: number;
    superseded_packets?: number;
    scores?: number[];
    classes?: string[];
    message?: string;
  };

  type PlaylistItem = { name: string; size: number; url: string; file?: File };
  type ScoreItem = { name: string; score: number; index: number };

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
  let thresholdPercent = 10;
  let camId = 'camera-01';
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
  let stopped = false;
  let seekWasPlaying = false;
  let suppressSeekRestart = false;
  let realtimeContext: AudioContext | null = null;
  let realtimeSource: MediaElementAudioSourceNode | null = null;
  let realtimeProcessor: ScriptProcessorNode | null = null;
  let realtimeSamples: number[] = [];
  let realtimePacketChain = Promise.resolve();

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
    { key: 'emergency sounds', aliases: ['siren', 'scream', 'alarm', 'ambulance', 'emergency'], title: 'Emergency Sounds', subtitle: 'Siren / Scream', color: '#ff4148', icon: Siren },
    { key: 'violence sounds', aliases: ['gunshot', 'gunfire', 'machine gun', 'battle cry', 'weapon', 'violence'], title: 'Violence Sounds', subtitle: 'Machine Gun / Gunshot / Battle Cry', color: '#8247ee', icon: Crosshair },
    { key: 'vehicle noise', aliases: ['engine', 'revving', 'race car', 'car horn', 'traffic', 'vehicle noise'], title: 'Vehicle Noise', subtitle: 'Modified Vehicle Noise / Engine Revving', color: '#5276f5', icon: CarFront },
    { key: 'glass breaking', aliases: ['glass', 'shatter'], title: 'Glass Breaking', subtitle: 'Glass Breaking', color: '#79e88d', icon: GlassWater },
    { key: 'impact sounds', aliases: ['impact', 'collision', 'crash', 'smash', 'thump', 'thud', 'heavy object drop'], title: 'Impact Sounds', subtitle: 'Vehicle Collision / Heavy Object Drop', color: '#f47c20', icon: Hammer },
    { key: 'explosion sounds', aliases: ['explosion', 'firecracker', 'firework', 'blast', 'detonation'], title: 'Explosion Sounds', subtitle: 'Explosion / Firecracker / Fireworks', color: '#ff3e43', icon: Bomb },
    { key: 'animal sounds', aliases: ['animal', 'dog', 'bark', 'canidae', 'wolf'], title: 'Animal Sounds', subtitle: 'Dog Bark', color: '#269957', icon: Dog }
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

    return { title, subtitle, color: colorFor(name), icon: AudioLines };
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
    const response = await fetch(path, options);
    if (!response.ok) {
      const body = await response.text();
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
    status = 'Playlist cleared';
  }

  async function selectItem(index: number, autoPlay = false): Promise<void> {
    if (index < 0 || index >= playlist.length) return;
    video?.pause();
    await stopStream();
    currentIndex = index;
    audioBuffer = null;
    scores = classes.map(() => 0);
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
    await api('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(message)
    });
  }

  function newStreamId(): number {
    requestCounter = (requestCounter + 1) % 1000;
    return Date.now() * 1000 + requestCounter;
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
        if (streamId === null) return;
        const input = event.inputBuffer;
        const channels = input.numberOfChannels;
        const frames = input.length;
        const first = input.getChannelData(0);
        const second = channels > 1 ? input.getChannelData(1) : first;
        for (let index = 0; index < frames; index += 1) {
          realtimeSamples.push(channels > 1 ? (first[index] + second[index]) * 0.5 : first[index]);
        }
        const id = streamId;
        while (realtimeSamples.length >= PACKET_SAMPLES && streamId === id) {
          const packet = realtimeSamples.splice(0, PACKET_SAMPLES);
          const timestamp = nextPacketTimestamp;
          nextPacketTimestamp += PACKET_MS;
          realtimePacketChain = realtimePacketChain.then(async () => {
            if (streamId !== id) return;
            await sendMessage({
              type: 'audio', id, cam_id: camId.trim(), timestamp_ms: timestamp,
              sample_rate: SAMPLE_RATE, channels: 1, encoding: 's16le',
              audio_b64: pcmFloatPacket(packet)
            });
          }).catch(showError);
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
    if (streamId !== null || decoding) return;
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
    await stopStream();
    if (!playlist.length) return;
    await selectItem((currentIndex + 1) % playlist.length);
    await startAnalysis(0, true);
  }

  async function pollEvents(): Promise<void> {
    while (!stopped) {
      try {
        const payload = await (await api(`/api/events?after=${eventSequence}`)).json();
        connected = true;
        eventSequence = payload.next ?? eventSequence;
        for (const envelope of payload.events ?? []) handleWorkerEvent(envelope.data as WorkerEvent);
      } catch (error) {
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
    if (event.event === 'ready') {
      classes = event.classes ?? [];
      scores = classes.map(() => 0);
      workerReady = classes.length > 0;
      status = `TensorRT worker ready · ${classes.length} aggregate classes`;
      statusKind = '';
      return;
    }
    if (event.event === 'worker_restarting') {
      workerReady = false;
      status = 'Restarting worker with saved mapping…';
      return;
    }
    if (event.event === 'fatal' || event.event === 'server_error') {
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
      scores = [...event.scores];
      const timestamp = event.timestamp_ms ?? 0;
      const lag = Math.max(0, Math.round(video.currentTime * 1000) - timestamp);
      status = `${event.cam_id} · ${timestamp} ms · inference ${event.processing_ms ?? 0} ms · playback lag ${lag} ms · superseded ${event.superseded_packets ?? 0}`;
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
  }

  onMount(() => {
    void loadMapping();
    void loadDefaultVideos();
    void pollEvents();
    pumpTimer = window.setInterval(() => void pumpPackets(), 20);
  });

  onDestroy(() => {
    stopped = true;
    if (pumpTimer !== undefined) window.clearInterval(pumpTimer);
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
  <div class="brand"><div class="mark">G</div><div><strong>GeoVision SED</strong><span>Sound Event Detection</span></div></div>
  <div class:online={connected && workerReady} class="connection">
    {connected ? (workerReady ? 'Worker ready' : 'Server connected') : 'Offline'}
  </div>
</header>

<main>
  <div class="studio">
    <div>
      <section class="panel preview-panel">
        <div class="panel-head"><h2>Media preview</h2><small>{currentItem?.name ?? 'No media selected'}</small></div>
        <div class="video-shell">
          <!-- svelte-ignore a11y_media_has_caption -->
          <video bind:this={video} src={currentItem?.url ?? ''} controls playsinline
            onplay={() => void handleNativePlay()} onseeking={handleSeeking} onseeked={() => void handleSeeked()} onended={() => void handleEnded()}></video>
          {#if !currentItem}<div class="empty-video">Choose one or more video/audio files</div>{/if}
        </div>
        <div class="controls-grid">
          <label>Camera ID<input bind:value={camId} placeholder="camera-01" /></label>
          <label>Alert threshold <span>{thresholdPercent.toFixed(1)}%</span><input type="range" min="0" max="100" step="0.5" bind:value={thresholdPercent} /></label>
          <label>Packet cadence<input value="40 ms" disabled /></label>
        </div>
        <div class="buttons">
          <input bind:this={fileInput} class="file-native" type="file" multiple accept="video/*,audio/*" onchange={addFiles} />
          <button class="primary" onclick={() => fileInput.click()}>Add media files</button>
          <button onclick={() => void runCurrent()} disabled={!currentItem || !workerReady || decoding}>Run current</button>
          <button class="danger" onclick={() => void clearPlaylist()} disabled={!playlist.length}>Clear</button>
        </div>
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
            {@const Icon = category.icon}
            <div class="triggered-card" style={`--label-color:${category.color}`}>
              <Icon size={31} strokeWidth={1.9} aria-hidden="true" />
              <div class="triggered-copy">
                <strong><b>{category.title}</b><b>{(item.score * 100).toFixed(1)}%</b></strong>
                <span>{category.subtitle}</span>
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
          {@const Icon = category.icon}
          <div class="score-row"
            style={`--label-color:${category.color};--score:${Math.min(100, item.score * 100)}%`}>
            <div class="category-icon" aria-hidden="true"><Icon size={34} strokeWidth={1.8} /></div>
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
          <textarea bind:value={mappingText} spellcheck="false" aria-label="Class mapping CSV"></textarea>
          <div class="buttons">
            <button class="primary" onclick={() => void saveMapping()}>Apply and save</button>
            <button onclick={() => void loadMapping()}>Reload saved</button>
            <input bind:this={mappingInput} class="file-native" type="file" accept=".csv,text/csv" onchange={(event) => void importMapping(event)} />
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
