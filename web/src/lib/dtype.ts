// Human explanations for tensor dtypes, shown in the dtype badge's hover bubble.

const INFO: Record<string, string> = {
  BF16: 'bfloat16 — 16-bit float (1 sign · 8 exponent · 7 mantissa). Same range as float32 with less precision; the common weights dtype for LLMs.',
  F16: 'float16 (IEEE half) — 16-bit float (1 · 5 · 10). More precision than bf16, but a much smaller range (±65504).',
  F32: 'float32 (IEEE single) — 32-bit float (1 · 8 · 23).',
  F64: 'float64 (IEEE double) — 64-bit float (1 · 11 · 52).',
  F8_E4M3: 'float8 E4M3 — 8-bit float (1 · 4 · 3). Higher precision, smaller range.',
  F8_E5M2: 'float8 E5M2 — 8-bit float (1 · 5 · 2). Larger range, less precision.',
  I8: 'int8 — 8-bit signed integer (two’s complement).',
  U8: 'uint8 — 8-bit unsigned integer.',
  I16: 'int16 — 16-bit signed integer.',
  U16: 'uint16 — 16-bit unsigned integer (also used as a packed-quantization container).',
  I32: 'int32 — 32-bit signed integer.',
  U32: 'uint32 — 32-bit unsigned integer.',
  I64: 'int64 — 64-bit signed integer.',
  U64: 'uint64 — 64-bit unsigned integer.',
  BOOL: 'bool — one byte per element.',
  U4: 'packed uint4 — 4-bit unsigned, two values per byte.',
  I4: 'packed int4 — 4-bit signed, two values per byte.',
};

export function dtypeInfo(dtype: string): string {
  return INFO[dtype.toUpperCase()] ?? `${dtype} — stored numeric type.`;
}
