<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { browserDecodeToWav } from './audio';
  export let triggered: string[] = [];
  export let active = false;
  export let duck: (speaking: boolean, volumePercent: number) => void;
  let duckPercent = 20;
  let speaking = false;
  $: duck(speaking, duckPercent);
  let enabled = false;
  let base = '';
  let sttModel = 'parakeet-tdt';
  let ttsModel = 'pocket-english';
  type SavedVoice = { id: string; name: string; transcript: string; audio: string; createdAt: number };
  let voice = 'woman';
  let voiceName = '';
  let voices: SavedVoice[] = [];
  let libraryBusy = false;
  let libraryGeneration = 0;
  let reference: File | null = null;
  let referenceUrl = '';
  let transcript = '';
  let previewText = 'Emergency Sounds';
  let status = '';
  let busy = false;
  let recording = false;
  let recorder: MediaRecorder | null = null;
  let stream: MediaStream | null = null;
  let controller: AbortController | null = null;
  let player: HTMLAudioElement | null = null;
  let finishPlayback: (() => void) | null = null;
  let previous = new Set<string>();
  let queue: string[] = [];
  let disposed = false;
  let alertSessionActive = false;
  $: observe(triggered, enabled, active);

  function observe(names: string[], on: boolean, playing: boolean) {
    const wasActive = alertSessionActive;
    alertSessionActive = on && playing;
    const next = new Set(alertSessionActive ? names : []);
    for (const name of next) if (!previous.has(name) && !queue.includes(name)) queue.push(name);
    previous = next;
    if (!alertSessionActive) { queue = []; if (wasActive) cancel(); }
    else void drain();
  }
  function cancel() {
    controller?.abort();
    player?.pause();
    finishPlayback?.();
    speaking = false;
  }
  async function api(path: string, options: RequestInit = {}) {
    const endpoint = (base.trim() || `${location.protocol}//${location.hostname}:8091/v1`).replace(/\/$/, '');
    const response = await fetch(`${endpoint}/${path}`, {
      ...options,
      signal: AbortSignal.any([AbortSignal.timeout(180000), ...(options.signal ? [options.signal] : [])])
    });
    if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
    return response;
  }
  function request(path: string, body: BodyInit, signal: AbortSignal, json = true) {
    return api(path, { method: 'POST', signal,
      headers: { 'Content-Type': json ? 'application/json' : 'audio/wav' }, body });
  }
  async function refreshVoices() {
    const generation = ++libraryGeneration;
    libraryBusy = true;
    try {
      const result = await (await api('ui/voices', { cache: 'no-store' })).json();
      if (generation !== libraryGeneration || disposed) return;
      voices = result.voices.sort((a: SavedVoice, b: SavedVoice) => a.name.localeCompare(b.name));
      if (voice.startsWith('saved:') && !voices.some(item => `saved:${item.id}` === voice)) {
        voice = 'woman';
        status = 'The selected shared voice was deleted. Using woman voice.';
      }
    } catch (error) { if (!disposed && generation === libraryGeneration) status = `Voice library: ${error}`; }
    finally { if (generation === libraryGeneration) libraryBusy = false; }
  }
  async function saveVoice() {
    if (!reference || !voiceName.trim() || busy || libraryBusy) return;
    busy = true;
    try {
      const audio = await browserDecodeToWav(reference, 24000, 1);
      if (audio.size > 16 * 1024 * 1024) throw new Error('Reference must be smaller than 16 MiB.');
      const encoded = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1]);
        reader.onerror = () => reject(new Error('Could not read reference audio.'));
        reader.readAsDataURL(audio);
      });
      const id = globalThis.crypto?.randomUUID?.() || `voice-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      await api('ui/voices', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, name: voiceName.trim(), transcript, audio: encoded, createdAt: Date.now() }) });
      await refreshVoices();
      voice = `saved:${id}`;
      status = `Created shared voice: ${voiceName.trim()}`;
    } catch (error) { if (!disposed) status = String(error); }
    finally { busy = false; }
  }
  async function deleteVoice() {
    if (!voice.startsWith('saved:') || busy || libraryBusy) return;
    const id = voice.slice(6);
    busy = true;
    try {
      await api(`ui/voices?id=${encodeURIComponent(id)}`, { method: 'DELETE' });
      voice = 'woman';
      await refreshVoices();
      status = 'Voice deleted from the shared library.';
    } catch (error) { if (!disposed) status = String(error); }
    finally { busy = false; }
  }
  function changeVoice() {
    cancel();
    queue = [];
  }
  function spokenLabel(label: string) {
    return label.replace(/\s*\(([^)]*)\)/g, '. $1').replace(/\//g, ', ').replace(/\s+/g, ' ').trim();
  }
  async function upload(file: File, signal: AbortSignal, rate: number) {
    const wav = await browserDecodeToWav(file, rate, 1);
    return (await (await request('ui/upload', wav, signal, false)).json()).path as string;
  }
  async function choose(file: File | null) {
    cancel();
    if (referenceUrl) URL.revokeObjectURL(referenceUrl);
    reference = file;
    referenceUrl = file ? URL.createObjectURL(file) : '';
    transcript = '';
    voice = file ? 'draft' : 'woman';
    voiceName = file?.name.replace(/\.[^.]+$/, '') || '';
    if (file) await transcribe();
  }
  async function transcribe() {
    if (!reference || busy) return;
    busy = true;
    const file = reference;
    const revision = transcript;
    const aborter = new AbortController(); controller = aborter;
    status = 'Transcribing reference…';
    try {
      const audio = await upload(file, aborter.signal, 16000);
      const result = await (await request('audio/transcriptions/details', JSON.stringify({ model: sttModel, audio }), aborter.signal)).json();
      if (!aborter.signal.aborted && reference === file && transcript === revision) transcript = result.text || '';
      status = 'Review the transcript to match the reference exactly.';
    } catch (error) { if (!aborter.signal.aborted) status = String(error); }
    finally { busy = false; }
  }
  async function speak(text: string) {
    const aborter = new AbortController(); controller = aborter;
    status = `Generating: ${text}`;
    const body: Record<string, unknown> = { model: ttsModel, input: spokenLabel(text), response_format: 'wav' };
    if (voice === 'woman' || voice === 'man') body.voice = voice;
    else {
      const saved = voices.find(item => `saved:${item.id}` === voice);
      const file = saved ? new File([Uint8Array.from(atob(saved.audio), char => char.charCodeAt(0))], 'reference.wav', { type: 'audio/wav' }) : reference;
      if (!file || (voice !== 'draft' && !saved)) throw new Error('Select a saved voice or record a reference first.');
      body.voice_ref = await upload(file, aborter.signal, 24000);
      body.reference_text = saved?.transcript ?? transcript;
    }
    const response = await request('audio/speech', JSON.stringify(body), aborter.signal);
    const blob = await response.blob();
    if (aborter.signal.aborted || disposed) return;
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url); player = audio;
    try {
      await new Promise<void>((resolve, reject) => {
        finishPlayback = resolve;
        audio.onended = () => resolve();
        audio.onerror = () => reject(new Error('Speech playback failed.'));
        audio.onplaying = () => { speaking = true; status = `Speaking: ${text}`; };
        void audio.play().catch(reject);
      });
      status = 'Ready';
    } finally {
      audio.pause(); player = null; finishPlayback = null;
      speaking = false; URL.revokeObjectURL(url);
    }
  }
  async function drain() {
    if (busy || disposed) return;
    busy = true;
    try { while (queue.length && enabled && active && !disposed) await speak(queue.shift()!); }
    catch (error) { if (!controller?.signal.aborted) status = String(error); queue = []; }
    finally { busy = false; }
  }
  async function preview() {
    if (busy) return;
    busy = true;
    try { await speak(previewText); }
    catch (error) { if (!controller?.signal.aborted) status = String(error); }
    finally { busy = false; }
  }
  async function record() {
    if (recording) { recorder?.stop(); return; }
    busy = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (disposed) { stream.getTracks().forEach(track => track.stop()); return; }
      const chunks: Blob[] = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        const type = recorder?.mimeType || 'audio/webm';
        stream?.getTracks().forEach(track => track.stop());
        recording = false; busy = false;
        if (!disposed) void choose(new File(chunks, 'reference.webm', { type }));
      };
      recorder.start(); recording = true;
    } catch (error) { stream?.getTracks().forEach(track => track.stop()); status = String(error); busy = false; }
  }
  onMount(() => {
    void refreshVoices();
    const refresh = () => { if (!busy) void refreshVoices(); };
    window.addEventListener('focus', refresh);
    return () => window.removeEventListener('focus', refresh);
  });
  onDestroy(() => {
    disposed = true; queue = []; cancel(); recorder?.state === 'recording' && recorder.stop();
    stream?.getTracks().forEach(track => track.stop());
    if (referenceUrl) URL.revokeObjectURL(referenceUrl);
  });
</script>

<details class="panel mapping-panel voice-panel" ontoggle={event => { if (event.currentTarget.open) void refreshVoices(); }}>
  <summary class="mapping-summary"><strong>Voice cloning & spoken alerts</strong></summary>
  <div class="mapping-content voice-content">
    <label class="alert-toggle"><input type="checkbox" bind:checked={enabled} /><span>Speak class names on threshold crossing</span></label>
    <label class="volume-control"><span>Media volume during speech <output>{duckPercent}%</output></span><input aria-label="Media volume during speech" type="range" min="0" max="100" step="1" bind:value={duckPercent} /></label>
    <div class="voice-fields">
      <label>Speech language<select aria-label="Speech language" bind:value={ttsModel} disabled={busy}>
        <option value="pocket-english">English</option>
        <option value="pocket-french">French</option>
      </select></label>
      <label>Voice<select aria-label="Voice" bind:value={voice} disabled={busy} onchange={changeVoice}>
        <option value="woman">Woman voice</option>
        <option value="man">Man voice</option>
        {#if reference}<option value="draft">Cloned voice — current reference</option>{/if}
        {#each voices as item (item.id)}<option value={`saved:${item.id}`}>Cloned voice — {item.name}</option>{/each}
      </select></label>
    </div>
    <div class="buttons library-actions">
      <button disabled={libraryBusy} onclick={() => void refreshVoices()}>Refresh voices</button>
      <button class="danger" disabled={busy || libraryBusy || !voice.startsWith('saved:')} onclick={() => void deleteVoice()}>Delete shared voice</button>
    </div>
    <details class="clone-editor" open>
      <summary>Create cloned voice</summary>
      <label>Reference audio<input type="file" accept="audio/*" disabled={busy} onchange={event => void choose(event.currentTarget.files?.[0] || null)} /></label>
      <div class="buttons recording-actions">
        <button class="microphone" class:recording aria-label={recording ? 'Stop recording' : 'Record voice'} title={recording ? 'Stop recording' : 'Record voice'} aria-pressed={recording} disabled={busy && !recording} onclick={() => void record()}>
          {#if recording}<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor" /></svg>
          {:else}<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3" /><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8" /></svg>{/if}
        </button>
        {#if recording}<span role="status">Recording…</span>{/if}
        <button disabled={!reference || busy} onclick={() => void transcribe()}>Transcribe reference</button>
      </div>
      {#if referenceUrl}<audio controls src={referenceUrl}></audio>{/if}
      <label>Reference transcript<textarea bind:value={transcript} rows="2"></textarea></label>
      <label>Voice name<input bind:value={voiceName} maxlength="100" placeholder="Name your cloned voice" /></label>
      <button disabled={busy || libraryBusy || !reference || !voiceName.trim()} onclick={() => void saveVoice()}>Create shared voice</button>
    </details>
    <label>Preview text<input bind:value={previewText} /></label>
    <div class="buttons">
      <button disabled={busy || (voice === 'draft' && !reference) || !previewText.trim()} onclick={() => void preview()}>Play TTS</button>
      <button onclick={cancel}>Cancel speech</button>
    </div>
    <details class="server-settings">
      <summary>Server settings</summary>
      <label>audio.cpp API URL<input bind:value={base} onchange={() => { voices = []; voice = 'woman'; void refreshVoices(); }} placeholder="http://hostname:8091/v1" disabled={busy} /></label>
      <label>STT model<input bind:value={sttModel} disabled={busy} /></label>
    </details>
    {#if status}<p class="voice-status" role="status">{status}</p>{/if}
  </div>
</details>

<style>
  .voice-content { display: grid; gap: 12px; padding-top: 14px; min-width: 0; }
  .voice-content label { min-width: 0; }
  .voice-content input, .voice-content select, .voice-content textarea { box-sizing: border-box; max-width: 100%; }
  .voice-content select { display: block; width: 100%; min-width: 0; margin-top: 5px; padding: 8px 10px; border: 1px solid #2a405f; border-radius: 8px; color: #edf5ff; background: #081426; font: inherit; }
  .voice-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 170px), 1fr)); gap: 12px; align-items: end; }
  .voice-content .alert-toggle { display: flex; align-items: center; gap: 9px; line-height: 1.5; cursor: pointer; }
  .voice-content .alert-toggle input { flex: 0 0 16px; width: 16px; height: 16px; margin: 0; padding: 0; accent-color: #55d9ca; }
  .volume-control > span { display: flex; justify-content: space-between; gap: 12px; }
  .volume-control input { display: block; height: 20px; }
  .voice-content .buttons { margin: 0; align-items: center; }
  .clone-editor { display: grid; min-width: 0; }
  .clone-editor > summary, .server-settings > summary { cursor: pointer; font-size: 12px; color: #b8cae4; }
  .clone-editor > :not(summary), .server-settings label { margin-top: 12px; }
  .voice-content textarea { min-height: 110px; height: 110px; }
  .voice-content audio { display: block; width: 100%; min-width: 0; }
  .microphone { display: inline-flex; justify-content: center; align-items: center; width: 38px; height: 38px; padding: 8px; flex-shrink: 0; }
  .microphone.recording { color: #ff566e; border-color: #ff566e; }
  .voice-status { margin: 0; overflow-wrap: anywhere; font-size: 12px; color: #b8cae4; }
</style>
