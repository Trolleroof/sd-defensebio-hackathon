// Decorative heart-rate trace for the hero: a resting band, an exertion ramp and recovery.
function buildPath(width: number, height: number) {
  const pts: string[] = [];
  const n = 240;
  for (let i = 0; i <= n; i++) {
    const x = (i / n) * width;
    const t = i / n;
    const load = t < 0.3 ? 0 : t < 0.62 ? 1 - Math.exp(-(t - 0.3) * 9) : Math.exp(-(t - 0.62) * 7) * 0.96;
    const beat = Math.sin(i * 0.31) * 0.025 + Math.sin(i * 0.11) * 0.02;
    const y = height * (0.82 - load * 0.62 - beat);
    pts.push(`${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  return pts.join(" ");
}

export function Trace() {
  const w = 1120;
  const h = 72;
  return (
    <svg className="trace" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} className="grid" x1={w * f} x2={w * f} y1="0" y2={h} />
      ))}
      <path d={buildPath(w, h)} pathLength={1} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
