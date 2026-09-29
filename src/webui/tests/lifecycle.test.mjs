import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

// Run the actual component controller with browser resources replaced by mocks.
// DOM rendering and browser media decoding require a separate browser smoke test.
const source = readFileSync(new URL('../src/routes/+page.svelte', import.meta.url), 'utf8');
const script = source.match(/<script lang="ts">([\s\S]*?)<\/script>/)[1]
  .replace(/^  import .*;\n/gm, '')
  .replace(/  \$: [\s\S]*?;\n/g, '');
const code = ts.transpile(script, { target: ts.ScriptTarget.ES2022 });
const deferred = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
};
const flush = () => new Promise(resolve => setImmediate(resolve));

function harness(fetchImpl = async () => ({ ok: true })) {
  let destroy;
  const messages = [];
  const revoked = [];
  const media = {
    paused: true, plays: 0, loads: 0, currentTime: 0,
    pause() { this.paused = true; },
    async play() { this.paused = false; this.plays++; },
    load() { this.loads++; }, removeAttribute() {}
  };
  const window = { setTimeout, clearTimeout, clearInterval, removeEventListener() {} };
  const controller = new Function('onMount', 'onDestroy', 'tick', 'displayNamesText', 'window', 'fetch', 'URL', 'media', 'messages', `
    let currentItem = null;
    ${code}
    video = media;
    workerReady = true;
    sendMessage = async message => { messages.push(message); };
    setupRealtimeAudio = async () => ({});
    return {
      startAnalysis, stopStream, clearPlaylist, loadDefaultVideos, pollEvents, api,
      maintainContinuousPlayback, handleSeeking, handleSeeked,
      selectItem, advancePlaylist, wakeWorkerIfStopped, stopDebugRun, StreamRequestError,
      enableDebugRun() { debugRunning = true; },
      setItems(items) { playlist = items; currentIndex = 0; currentItem = items[0]; },
      changePlaylist(items) { playlistVersion++; playlist = items; currentItem = items[0]; },
      setSender(fn) { sendMessage = fn; },
      setSetup(fn) { setupRealtimeAudio = fn; },
      state() { return { playlist, streamId, stopped, starting, connected, resumeAfterWorkerRestart, debugRunning, workerReady }; }
    };
  `)(() => {}, fn => { destroy = fn; }, async () => {}, '', window, fetchImpl,
    { revokeObjectURL: url => revoked.push(url) }, media, messages);
  return { ...controller, destroy: () => destroy(), media, messages, revoked };
}
const upload = { name: 'local.mp4', url: 'blob:local', size: 123, file: {} };

test('uploaded media streams without reading or decoding the entire file', async () => {
  const h = harness();
  h.setItems([upload]);
  await h.startAnalysis();
  assert.equal(h.media.plays, 1);
  assert.equal(h.messages[0].type, 'stream_start');
  await h.stopStream();
  assert.equal(h.messages[1].type, 'stream_end');
});

test('clear invalidates a start waiting for audio setup and revokes the upload', async () => {
  const h = harness();
  const setup = deferred();
  h.setItems([upload]);
  h.setSetup(() => setup.promise);
  const start = h.startAnalysis();
  await flush();
  await h.clearPlaylist();
  setup.resolve({});
  await start;
  assert.equal(h.media.plays, 0);
  assert.equal(h.state().streamId, null);
  assert.deepEqual(h.state().playlist, []);
  assert.deepEqual(h.revoked, ['blob:local']);
});

test('overlapping starts only start the newest operation', async () => {
  const h = harness();
  const setup = deferred();
  h.setItems([upload]);
  h.setSetup(() => setup.promise);
  const first = h.startAnalysis();
  await flush();
  const second = h.startAnalysis();
  await flush();
  setup.resolve({});
  await Promise.all([first, second]);
  assert.equal(h.media.plays, 1);
  assert.equal(h.messages.filter(m => m.type === 'stream_start').length, 1);
});

test('late default playlist response preserves uploaded files', async () => {
  const response = deferred();
  const h = harness(() => response.promise);
  const loading = h.loadDefaultVideos();
  h.changePlaylist([upload]);
  response.resolve({ ok: true, json: async () => ({ videos: [{ name: 'default', url: '/default' }] }) });
  await loading;
  assert.deepEqual(h.state().playlist, [upload]);
  h.destroy();
  assert.deepEqual(h.revoked, ['blob:local']);
});

test('destroy invalidates pending audio setup', async () => {
  const h = harness();
  const setup = deferred();
  h.setItems([upload]);
  h.setSetup(() => setup.promise);
  const start = h.startAnalysis();
  await flush();
  h.destroy();
  setup.resolve({});
  await start;
  assert.equal(h.media.plays, 0);
  assert.equal(h.messages.length, 0);
});

test('destroy aborts requests and ignores a late event response', async () => {
  const response = deferred();
  let signal;
  const h = harness((_path, options) => { signal = options.signal; return response.promise; });
  const poll = h.pollEvents();
  h.destroy();
  assert.equal(signal.aborted, true);
  response.resolve({ ok: true, json: async () => ({ next: 1, events: [{ data: { event: 'ready', classes: ['sound'] } }] }) });
  await poll;
  assert.equal(h.state().connected, false);
  await assert.rejects(h.api('/api/health'), { name: 'AbortError' });
});


test('stop waits for an in-flight stream start before sending stream end', async () => {
  const h = harness();
  const request = deferred();
  const messages = [];
  h.setItems([upload]);
  h.setSender(async message => {
    messages.push(message.type);
    if (message.type === 'stream_start') await request.promise;
  });
  const start = h.startAnalysis();
  await flush();
  const stop = h.stopStream();
  await flush();
  assert.deepEqual(messages, ['stream_start']);
  request.resolve();
  await Promise.all([start, stop]);
  assert.deepEqual(messages, ['stream_start', 'stream_end']);
  assert.equal(h.media.plays, 0);
});


test('transient stream-start failure during playlist advance preserves the continuous run', async () => {
  const h = harness();
  h.setItems([upload]);
  h.enableDebugRun();
  h.setSender(async message => {
    if (message.type === 'stream_start') throw new h.StreamRequestError('temporary timeout');
  });
  await h.advancePlaylist();
  assert.equal(h.state().debugRunning, true);
  assert.equal(h.state().resumeAfterWorkerRestart, true);
  assert.equal(h.state().workerReady, false);
});

test('failed recovery is retried on subsequent health polls', async () => {
  const h = harness(async () => ({ ok: true, json: async () => ({ worker_ready: true, classes: ['sound'] }) }));
  h.setItems([upload]);
  let attempts = 0;
  h.setSender(async message => {
    if (message.type === 'stream_start' && ++attempts <= 2) throw new h.StreamRequestError('temporary timeout');
  });
  await assert.rejects(h.startAnalysis());
  await h.wakeWorkerIfStopped();
  await flush();
  assert.equal(h.state().resumeAfterWorkerRestart, true);
  await h.wakeWorkerIfStopped();
  await flush();
  assert.equal(attempts, 3);
  assert.equal(h.media.paused, false);
  assert.equal(h.state().resumeAfterWorkerRestart, false);
});

test('explicit stop cancels scheduled automatic recovery', async () => {
  const h = harness(async () => ({ ok: true, json: async () => ({ worker_ready: true, classes: ['sound'] }) }));
  h.setItems([upload]);
  h.setSender(async message => {
    if (message.type === 'stream_start') throw new h.StreamRequestError('temporary timeout');
  });
  await assert.rejects(h.startAnalysis());
  await h.stopDebugRun();
  await h.wakeWorkerIfStopped();
  await flush();
  assert.equal(h.state().resumeAfterWorkerRestart, false);
  assert.equal(h.media.plays, 0);
});


test('browser autoplay rejection is not retried as a server outage', async () => {
  const h = harness();
  h.setItems([upload]);
  h.media.play = async () => { throw new DOMException('Autoplay blocked', 'NotAllowedError'); };
  await assert.rejects(h.startAnalysis(), { name: 'NotAllowedError' });
  assert.equal(h.state().resumeAfterWorkerRestart, false);
  assert.equal(h.state().streamId, null);
});


test('continuous mode resumes an unexplained pause without recreating the stream', async () => {
  const h = harness();
  h.setItems([upload]);
  await h.startAnalysis();
  h.enableDebugRun();
  const id = h.state().streamId;
  h.media.pause();
  await h.maintainContinuousPlayback();
  assert.equal(h.media.paused, false);
  assert.equal(h.state().streamId, id);
  assert.equal(h.messages.filter(m => m.type === 'stream_start').length, 1);
});

test('continuous playback watchdog respects Stop and Clear', async () => {
  for (const action of ['stopDebugRun', 'clearPlaylist']) {
    const h = harness();
    h.setItems([upload]);
    h.enableDebugRun();
    await h.startAnalysis();
    await h[action]();
    await h.maintainContinuousPlayback();
    assert.equal(h.media.paused, true);
    assert.equal(h.media.plays, 1);
  }
});

test('continuous playback watchdog leaves seeking and ended media alone', async () => {
  for (const state of ['seeking', 'ended']) {
    const h = harness();
    h.setItems([upload]);
    h.enableDebugRun();
    h.media[state] = true;
    await h.maintainContinuousPlayback();
    assert.equal(h.media.plays, 0);
  }
});

test('seek from a temporary pause preserves continuous playback intent', async () => {
  const h = harness();
  h.setItems([upload]);
  await h.startAnalysis();
  h.enableDebugRun();
  h.media.pause();
  h.handleSeeking();
  await h.handleSeeked();
  assert.equal(h.media.paused, false);
});

test('rejected continuous resume is throttled and does not overlap', async () => {
  const h = harness();
  h.setItems([upload]);
  await h.startAnalysis();
  h.enableDebugRun();
  h.media.pause();
  let attempts = 0;
  h.media.play = async () => { attempts++; throw new Error('temporarily blocked'); };
  await Promise.all([h.maintainContinuousPlayback(), h.maintainContinuousPlayback()]);
  await h.maintainContinuousPlayback();
  assert.equal(attempts, 1);
});
