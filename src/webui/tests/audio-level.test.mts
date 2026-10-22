import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeAudioLevel } from '../src/components/audio.ts';

function buffer(amplitudes: number[], frames = 3200): AudioBuffer {
  const channels = amplitudes.map(amplitude => Float32Array.from({ length: frames }, (_, i) => amplitude * Math.sin(2 * Math.PI * 100 * i / 16000)));
  return { length: frames, sampleRate: 16000, numberOfChannels: channels.length, getChannelData: (channel: number) => channels[channel] } as unknown as AudioBuffer;
}
function rms(samples: Float32Array): number {
  return Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
}
function peak(samples: Float32Array): number {
  return samples.reduce((maximum, sample) => Math.max(maximum, Math.abs(sample)), 0);
}

test('quiet and loud voices reach the same playback RMS target', () => {
  const quiet = normalizeAudioLevel(buffer([0.03]));
  const loud = normalizeAudioLevel(buffer([0.6]));
  const target = Math.pow(10, -16 / 20);
  assert.ok(Math.abs(rms(quiet.getChannelData(0)) - target) < 0.000001);
  assert.ok(Math.abs(rms(loud.getChannelData(0)) - target) < 0.000001);
});

test('reference recordings reach the common target without altering duration', () => {
  const reference = buffer([0.03]);
  assert.equal(normalizeAudioLevel(reference), reference);
  assert.ok(Math.abs(rms(reference.getChannelData(0)) - Math.pow(10, -16 / 20)) < 0.000001);
  assert.equal(reference.length, 3200);
  assert.equal(reference.sampleRate, 16000);
});

test('pauses do not cause speech to be over-amplified', () => {
  const reference = buffer([0.03], 6400);
  reference.getChannelData(0).fill(0, 3200);
  normalizeAudioLevel(reference);
  assert.ok(Math.abs(rms(reference.getChannelData(0).subarray(0, 3200)) - Math.pow(10, -16 / 20)) < 0.000001);
  assert.ok(reference.getChannelData(0).subarray(3200).every(sample => sample === 0));
});

test('silence and near-silent noise are not amplified', () => {
  for (const amplitude of [0, 0.0001]) {
    const reference = buffer([amplitude]);
    const before = reference.getChannelData(0).slice();
    normalizeAudioLevel(reference);
    assert.deepEqual(reference.getChannelData(0), before);
  }
});

test('transient peaks retain headroom instead of clipping', () => {
  const reference = buffer([0.01]);
  reference.getChannelData(0)[100] = 1;
  normalizeAudioLevel(reference);
  assert.ok(peak(reference.getChannelData(0)) <= 0.950001);
  assert.ok(reference.getChannelData(0).every(Number.isFinite));
});

test('gain is bounded for very quiet recordings', () => {
  const reference = buffer([0.0016]);
  const before = rms(reference.getChannelData(0));
  normalizeAudioLevel(reference);
  assert.ok(Math.abs(rms(reference.getChannelData(0)) / before - 16) < 0.00001);
});

test('stereo channels share one gain and preserve their balance', () => {
  const reference = buffer([0.1, 0.05]);
  normalizeAudioLevel(reference);
  assert.equal(reference.numberOfChannels, 2);
  assert.ok(Math.abs(rms(reference.getChannelData(0)) / rms(reference.getChannelData(1)) - 2) < 0.00001);
});

test('reusing a normalized reference does not keep increasing its volume', () => {
  const reference = buffer([0.03]);
  normalizeAudioLevel(reference);
  const before = reference.getChannelData(0).slice();
  normalizeAudioLevel(reference);
  assert.ok(reference.getChannelData(0).every((sample, index) => Math.abs(sample - before[index]) < 0.000001));
});
