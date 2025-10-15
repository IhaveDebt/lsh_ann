/**
 * Simple LSH-based Approximate Nearest Neighbors (lsh_ann.ts)
 *
 * Uses random hyperplane hashing for cosine similarity approx.
 * Run: ts-node src/lsh_ann.ts
 */

type Vec = number[];

function dot(a: Vec, b: Vec) { return a.reduce((s, v, i) => s + v * b[i], 0); }
function norm(a: Vec) { return Math.sqrt(a.reduce((s, v) => s + v * v, 0)); }
function cosine(a: Vec, b: Vec) { return dot(a, b) / (norm(a) * norm(b) + 1e-12); }

class LSH {
  bands: number;
  dim: number;
  hashes: Vec[]; // random hyperplanes

  index: Map<string, number[]> = new Map();
  points: Vec[] = [];

  constructor(dim: number, bands = 16) {
    this.dim = dim;
    this.bands = bands;
    this.hashes = Array.from({ length: bands }, () => Array.from({ length: dim }, () => Math.random() * 2 - 1));
  }

  private signature(v: Vec) {
    return this.hashes.map(h => (dot(h, v) >= 0 ? '1' : '0')).join('');
  }

  add(v: Vec) {
    const id = this.points.length;
    this.points.push(v);
    const sig = this.signature(v);
    for (let i = 0; i < this.bands; i += 4) {
      const bucket = sig.slice(i, i + 4);
      const key = `${i}:${bucket}`;
      this.index.set(key, (this.index.get(key) || []).concat(id));
    }
  }

  query(q: Vec, topK = 5) {
    const sig = this.signature(q);
    const candidates = new Set<number>();
    for (let i = 0; i < this.bands; i += 4) {
      const bucket = sig.slice(i, i + 4);
      const key = `${i}:${bucket}`;
      (this.index.get(key) || []).forEach(id => candidates.add(id));
    }
    const scored = Array.from(candidates).map(id => ({ id, score: cosine(q, this.points[id]) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }
}

// demo
if (require.main === module) {
  const dim = 32;
  const lsh = new LSH(dim, 32);
  for (let i = 0; i < 200; i++) {
    lsh.add(Array.from({ length: dim }, () => Math.random()));
  }
  const q = lsh.points[Math.floor(Math.random() * lsh.points.length)];
  console.log('Query nearest (approx):', lsh.query(q, 5));
}
