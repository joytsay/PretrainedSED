function writeAscii(view: DataView, offset: number, text: string) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

export function encodePcm16Wav(buffer: AudioBuffer): Blob {
  const channels = buffer.numberOfChannels;
  const frames = buffer.length;
  const bytes = new ArrayBuffer(44 + frames * channels * 2);
  const view = new DataView(bytes);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + frames * channels * 2, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, frames * channels * 2, true);

  const samples = Array.from({ length: channels }, (_, channel) => buffer.getChannelData(channel));
  let offset = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, samples[channel][frame]));
      view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
      offset += 2;
    }
  }
  return new Blob([bytes], { type: 'audio/wav' });
}

export async function concatenateAudioBlobs(blobs: Blob[]): Promise<Blob> {
  if (!blobs.length) throw new Error('No audio chunks were generated.');
  if (blobs.length === 1) return blobs[0];

  const context = new AudioContext();
  try {
    const decoded = await Promise.all(blobs.map(async (blob) =>
      context.decodeAudioData(await blob.arrayBuffer())));
    const sampleRate = decoded[0].sampleRate;
    const channels = decoded[0].numberOfChannels;
    for (const chunk of decoded) {
      if (chunk.sampleRate !== sampleRate || chunk.numberOfChannels !== channels) {
        throw new Error('Generated chunks use different audio formats and cannot be joined.');
      }
    }
    const totalFrames = decoded.reduce((sum, chunk) => sum + chunk.length, 0);
    const merged = context.createBuffer(channels, totalFrames, sampleRate);
    let offset = 0;
    for (const chunk of decoded) {
      for (let channel = 0; channel < channels; channel += 1) {
        merged.copyToChannel(chunk.getChannelData(channel), channel, offset);
      }
      offset += chunk.length;
    }
    return encodePcm16Wav(merged);
  } finally {
    await context.close();
  }
}

export async function browserDecodeToWav(
  file: File,
  targetSampleRate?: number,
  targetChannels?: number
): Promise<Blob> {
  if (!targetSampleRate && !targetChannels && (file.type === 'audio/wav' || file.name.toLowerCase().endsWith('.wav'))) {
    return file;
  }
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const sampleRate = targetSampleRate || decoded.sampleRate;
    const channels = targetChannels || decoded.numberOfChannels;
    if (decoded.sampleRate === sampleRate && decoded.numberOfChannels === channels) {
      return encodePcm16Wav(decoded);
    }
    const frames = Math.max(1, Math.ceil(decoded.duration * sampleRate));
    const offline = new OfflineAudioContext(channels, frames, sampleRate);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    return encodePcm16Wav(await offline.startRendering());
  } finally {
    await context.close();
  }
}

/** Normalize speech-active RMS, excluding pauses and retaining peak headroom. */
export function normalizeAudioLevel(buffer: AudioBuffer, targetDb = -16): AudioBuffer {
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, channel) => buffer.getChannelData(channel));
  const windowFrames = Math.max(1, Math.round(buffer.sampleRate * 0.02));
  const windows: Array<{ power: number; count: number; rms: number }> = [];
  let peak = 0;
  let maximumRms = 0;
  for (let start = 0; start < buffer.length; start += windowFrames) {
    const end = Math.min(buffer.length, start + windowFrames);
    let power = 0;
    for (const samples of channels) {
      for (let frame = start; frame < end; frame += 1) {
        const sample = samples[frame];
        peak = Math.max(peak, Math.abs(sample));
        power += sample * sample;
      }
    }
    const count = (end - start) * channels.length;
    const rms = Math.sqrt(power / count);
    maximumRms = Math.max(maximumRms, rms);
    windows.push({ power, count, rms });
  }
  // Do not amplify silence or a near-silent microphone's noise floor.
  const gate = Math.max(0.001, maximumRms * 0.1);
  let speechPower = 0;
  let speechSamples = 0;
  for (const window of windows) {
    if (window.rms < gate) continue;
    speechPower += window.power;
    speechSamples += window.count;
  }
  if (!speechSamples || !peak) return buffer;
  const rms = Math.sqrt(speechPower / speechSamples);
  const gain = Math.min(16, Math.pow(10, targetDb / 20) / rms, 0.95 / peak);
  for (const samples of channels) {
    for (let frame = 0; frame < samples.length; frame += 1) samples[frame] *= gain;
  }
  return buffer;
}

export async function normalizeSpeechVolume(audio: Blob, targetDb = -16): Promise<Blob> {
  const decoder = new OfflineAudioContext(1, 1, 24000);
  const decoded = await decoder.decodeAudioData(await audio.arrayBuffer());
  return encodePcm16Wav(normalizeAudioLevel(decoded, targetDb));
}

export async function normalizeReferenceVoice(file: File): Promise<File> {
  const wav = await browserDecodeToWav(file, 24000, 1);
  const normalized = await normalizeSpeechVolume(wav);
  return new File([normalized], `${file.name.replace(/\.[^.]+$/, '')}.wav`, { type: 'audio/wav' });
}
