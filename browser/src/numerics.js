const float = new Float32Array(1),
  bits = new Uint32Array(float.buffer);
export function half(value) {
  float[0] = value;
  const x = bits[0],
    sign = (x >>> 16) & 0x8000,
    exp = ((x >>> 23) & 255) - 127 + 15,
    mantissa = x & 0x7fffff;
  if (exp >= 31) return sign | 0x7c00 | (mantissa && exp === 143 ? 0x200 : 0);
  if (exp <= 0) {
    if (exp < -10) return sign;
    const m = mantissa | 0x800000,
      shift = 14 - exp;
    return (
      sign | ((m + (1 << (shift - 1)) - 1 + ((m >>> shift) & 1)) >>> shift)
    );
  }
  return (
    sign | ((exp << 10) + ((mantissa + 0xfff + ((mantissa >>> 13) & 1)) >>> 13))
  );
}
export function toHalf(values) {
  const result = new Uint16Array(values.length);
  for (let i = 0; i < result.length; i++) result[i] = half(values[i]);
  return result;
}
export function noise(seed, n) {
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-9))),
      angle = 2 * Math.PI * random();
    out[i] = radius * Math.cos(angle);
    if (i + 1 < n) out[i + 1] = radius * Math.sin(angle);
  }
  return out;
}
export function dpmStep(sample, epsilon, sigmas, index, previous) {
  const s = sigmas[index],
    next = sigmas[index + 1],
    alpha = 1 / Math.sqrt(s * s + 1),
    sigma = s * alpha,
    alphaNext = 1 / Math.sqrt(next * next + 1),
    sigmaNext = next * alphaNext;
  const clean = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++)
    clean[i] = (sample[i] - sigma * epsilon[i]) / alpha;
  if (next === 0) return { sample: clean, previous: clean };
  const h = Math.log(s / next),
    coefficient = -alphaNext * Math.expm1(-h),
    ratio = sigmaNext / sigma;
  const r = index ? Math.log(sigmas[index - 1] / s) / h : 1;
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++)
    out[i] =
      ratio * sample[i] +
      coefficient *
        (clean[i] + (previous ? (0.5 * (clean[i] - previous[i])) / r : 0));
  return { sample: out, previous: clean };
}
