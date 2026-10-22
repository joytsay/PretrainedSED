<script lang="ts">
  import { onDestroy } from 'svelte';
  import { browserDecodeToWav } from './audio';
  export let triggered: string[] = [];
  export let active = false;
  export let duck: (speaking: boolean) => void;
  let enabled = false;
  let base = '';
  let sttModel = 'parakeet-tdt';
  let ttsModel = 'pocket-english';
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
  $: observe(triggered, enabled, active);

  function observe(names: string[], on: boolean, playing: boolean) {
    const next = new Set(on && playing ? names : []);
    for (const name of next) if (!previous.has(name) && !queue.includes(name)) queue.push(name);
    previous = next;
    if (!on || !playing) { queue = []; cancel(); }
    else void drain();
  }
  function cancel() {
    controller?.abort();
    player?.pause();
    finishPlayback?.();
    duck(false);
  }
  async function request(path: string, body: BodyInit, signal: AbortSignal, json = true) {
    const endpoint = (base.trim() || `${location.protocol}//${location.hostname}:8091/v1`).replace(/\/$/, '');
    const response = await fetch(`${endpoint}/${path}`, {
      method: 'POST', signal, headers: { 'Content-Type': json ? 'application/json' : 'audio/wav' }, body
    });
    if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
    return response;
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
    if (!reference) throw new Error('Choose reference audio first.');
    const aborter = new AbortController(); controller = aborter;
    status = `Generating: ${text}`;
    const voice_ref = await upload(reference, aborter.signal, 24000);
    const response = await request('audio/speech', JSON.stringify({ model: ttsModel, input: text, response_format: 'wav', voice_ref, reference_text: transcript }), aborter.signal);
    const blob = await response.blob();
    if (aborter.signal.aborted || disposed) return;
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url); player = audio;
    try {
      await new Promise<void>((resolve, reject) => {
        finishPlayback = resolve;
        audio.onended = () => resolve();
        audio.onerror = () => reject(new Error('Speech playback failed.'));
        audio.onplaying = () => { duck(true); status = `Speaking: ${text}`; };
        void audio.play().catch(reject);
      });
      status = 'Ready';
    } finally {
      audio.pause(); player = null; finishPlayback = null;
      duck(false); URL.revokeObjectURL(url);
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
  onDestroy(() => {
    disposed = true; queue = []; cancel(); recorder?.state === 'recording' && recorder.stop();
    stream?.getTracks().forEach(track => track.stop());
    if (referenceUrl) URL.revokeObjectURL(referenceUrl);
  });
</script>

<details class="panel mapping-panel">
  <summary class="mapping-summary"><strong>Voice cloning & spoken alerts</strong></summary>
  <div class="mapping-content">
    <label><input type="checkbox" bind:checked={enabled} disabled={!reference} /> Speak class names on threshold crossing</label>
    <p>Each class is read once per trigger. Media plays at 20% volume during speech and returns afterward.</p>
    <label>audio.cpp API URL<input bind:value={base} placeholder="http://hostname:8091/v1" disabled={busy} /></label>
    <label>STT model<input bind:value={sttModel} disabled={busy} /></label>
    <label>Speech language<select bind:value={ttsModel} disabled={busy}>
      <option value="pocket-english">English — PocketTTS</option>
      <option value="pocket-french">French — PocketTTS</option>
    </select></label>
    <p>Choose the language of the class names or preview text. Labels are spoken as written, without translation. PocketTTS clones from reference audio; its transcript is optional.</p>
    <label>Reference voice<input type="file" accept="audio/*" disabled={busy} onchange={event => void choose(event.currentTarget.files?.[0] || null)} /></label>
    <div class="buttons">
      <button disabled={busy && !recording} onclick={() => void record()}>{recording ? 'Stop recording' : 'Record voice'}</button>
      <button disabled={!reference || busy} onclick={() => void transcribe()}>Transcribe reference</button>
      <button onclick={cancel}>Cancel speech</button>
    </div>
    {#if referenceUrl}<audio controls src={referenceUrl}></audio>{/if}
    <label>Reference transcript<textarea bind:value={transcript} rows="3"></textarea></label>
    <label>Preview text<input bind:value={previewText} /></label>
    <button disabled={busy || !reference || !previewText.trim()} onclick={() => void preview()}>Play TTS</button>
    <p role="status">{status}</p>
  </div>
</details>
