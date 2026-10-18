"use client";

import { HORIZON } from "@/lib/data";
import { useInView } from "@/lib/useInView";

const W = 620;
const H = 240;
const PAD = { l: 36, r: 12, t: 12, b: 28 };
const Y_MAX = 12;

function x(i: number) {
  return PAD.l + (i / (HORIZON.seconds.length - 1)) * (W - PAD.l - PAD.r);
}
function y(v: number) {
  return PAD.t + (1 - v / Y_MAX) * (H - PAD.t - PAD.b);
}
function path(values: number[]) {
  return values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
}

export function HorizonChart() {
  const [ref, inView] = useInView<SVGSVGElement>(0.3);
  return (
    <>
      <div className="legend">
        <span><i style={{ background: "var(--olive)" }} />model-a</span>
        <span><i style={{ background: "var(--faint)" }} />kinetic_pop_calibrated</span>
      </div>
      <svg
        ref={ref}
        className={`chart ${inView ? "in" : ""}`}
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Placeholder: transfer MAE by forecast horizon"
      >
        {[0, 4, 8, 12].map((v) => (
          <g key={v}>
            <line className="axis" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
            <text x={PAD.l - 8} y={y(v) + 3} textAnchor="end">{v}</text>
          </g>
        ))}
        {HORIZON.seconds.map((s, i) =>
          i % 2 === 1 ? (
            <text key={s} x={x(i)} y={H - 8} textAnchor="middle">{s}s</text>
          ) : null,
        )}
        <path className="line anchor" d={path(HORIZON.anchor)} pathLength={1} />
        <path className="line agent" d={path(HORIZON.agent)} pathLength={1} />
      </svg>
    </>
  );
}
