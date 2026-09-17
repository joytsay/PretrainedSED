export const MEL_BANDS = 64;
export const FFT_SIZE = 1024;

const MAGMA_STOPS = [
  '#000004', '#1b0c41', '#4f0c6b', '#781c6d', '#a52c60',
  '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcfdbf'
];

function fft(real: Float64Array, imaginary: Float64Array): void {
  const size = real.length;
  for (let index = 1, reversed = 0; index < size; index += 1) {
    let bit = size >> 1;
    for (; reversed & bit; bit >>= 1) reversed ^= bit;
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed], real[index]];
      [imaginary[index], imaginary[reversed]] = [imaginary[reversed], imaginary[index]];
    }
  }

  for (let length = 2; length <= size; length <<= 1) {
    const angle = -2 * Math.PI / length;
    const stepReal = Math.cos(angle);
    const stepImaginary = Math.sin(angle);
    for (let offset = 0; offset < size; offset += length) {
      let rotationReal = 1;
      let rotationImaginary = 0;
      for (let index = 0; index < length / 2; index += 1) {
        const even = offset + index;
        const odd = even + length / 2;
        const oddReal = real[odd] * rotationReal - imaginary[odd] * rotationImaginary;
        const oddImaginary = real[odd] * rotationImaginary + imaginary[odd] * rotationReal;
        real[odd] = real[even] - oddReal;
        imaginary[odd] = imaginary[even] - oddImaginary;
        real[even] += oddReal;
        imaginary[even] += oddImaginary;
        const nextReal = rotationReal * stepReal - rotationImaginary * stepImaginary;
        rotationImaginary = rotationReal * stepImaginary + rotationImaginary * stepReal;
        rotationReal = nextReal;
      }
    }
  }
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

/**
 * Compute one normalized 64-band mel column using the same primary settings as
 * render_sed_overlay_video.py: 16 kHz audio, a 1024-sample Hann window, and a
 * 60-7800 Hz mel range. Values are normalized over an 80 dB display range.
 */
export function computeMelColumn(samples: Float32Array, sampleRate: number): Float32Array {
  const real = new Float64Array(FFT_SIZE);
  const imaginary = new Float64Array(FFT_SIZE);
  const sourceOffset = Math.max(0, samples.length - FFT_SIZE);
  const destinationOffset = Math.max(0, FFT_SIZE - samples.length);
  for (let index = destinationOffset; index < FFT_SIZE; index += 1) {
    const sample = samples[sourceOffset + index - destinationOffset] ?? 0;
    real[index] = sample * (0.5 - 0.5 * Math.cos(2 * Math.PI * index / FFT_SIZE));
  }
  fft(real, imaginary);

  const power = new Float64Array(FFT_SIZE / 2 + 1);
  for (let bin = 0; bin < power.length; bin += 1) {
    power[bin] = real[bin] ** 2 + imaginary[bin] ** 2;
  }

  const minimumMel = hzToMel(60);
  const maximumMel = hzToMel(Math.min(7800, sampleRate / 2));
  const boundaries = new Float64Array(MEL_BANDS + 2);
  for (let index = 0; index < boundaries.length; index += 1) {
    const mel = minimumMel + (maximumMel - minimumMel) * index / (MEL_BANDS + 1);
    boundaries[index] = melToHz(mel) * FFT_SIZE / sampleRate;
  }

  const decibels = new Float64Array(MEL_BANDS);
  let maximumDb = -Infinity;
  for (let band = 0; band < MEL_BANDS; band += 1) {
    const left = boundaries[band];
    const center = boundaries[band + 1];
    const right = boundaries[band + 2];
    let energy = 0;
    for (let bin = Math.max(0, Math.floor(left)); bin <= Math.min(power.length - 1, Math.ceil(right)); bin += 1) {
      const weight = bin <= center
        ? (bin - left) / Math.max(center - left, Number.EPSILON)
        : (right - bin) / Math.max(right - center, Number.EPSILON);
      energy += power[bin] * Math.max(0, weight);
    }
    decibels[band] = 10 * Math.log10(Math.max(energy, 1e-12));
    maximumDb = Math.max(maximumDb, decibels[band]);
  }

  const normalized = new Float32Array(MEL_BANDS);
  if (maximumDb <= -100) return normalized;
  for (let band = 0; band < MEL_BANDS; band += 1) {
    normalized[band] = Math.max(0, Math.min(1, (decibels[band] - maximumDb + 80) / 80));
  }
  return normalized;
}

function parseHex(color: string): [number, number, number] {
  const value = Number.parseInt(color.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function magmaColor(value: number): string {
  const scaled = Math.max(0, Math.min(1, value)) * (MAGMA_STOPS.length - 1);
  const lower = Math.floor(scaled);
  const upper = Math.min(MAGMA_STOPS.length - 1, lower + 1);
  const fraction = scaled - lower;
  const start = parseHex(MAGMA_STOPS[lower]);
  const end = parseHex(MAGMA_STOPS[upper]);
  const channel = (index: number) => Math.round(start[index] + (end[index] - start[index]) * fraction);
  return `rgb(${channel(0)},${channel(1)},${channel(2)})`;
}
