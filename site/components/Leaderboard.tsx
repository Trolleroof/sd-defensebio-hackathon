"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { ENTRIES, METRICS, type MetricKey } from "@/lib/data";
import { useInView } from "@/lib/useInView";
import { CountUp } from "./CountUp";

const ORDER: MetricKey[] = ["pass", "mae", "judge", "cost"];

export function Leaderboard() {
  const [metric, setMetric] = useState<MetricKey>("pass");
  const [ref, inView] = useInView<HTMLDivElement>(0.15);
  const rowRefs = useRef(new Map<string, HTMLDivElement>());
  const lastTops = useRef(new Map<string, number>());

  const m = METRICS[metric];
  const scored = ENTRIES.filter((e) => !e.baseline).sort((a, b) =>
    m.lowerIsBetter ? a[metric] - b[metric] : b[metric] - a[metric],
  );
  // The anchor only has a meaningful value for transfer error; elsewhere it sits last.
  const anchor = ENTRIES.filter((e) => e.baseline);
  const rows = [...scored, ...anchor];

  // FLIP: slide rows from their previous position when the sort order changes.
  useLayoutEffect(() => {
    rowRefs.current.forEach((el, key) => {
      const top = el.getBoundingClientRect().top;
      const prev = lastTops.current.get(key);
      if (prev !== undefined && prev !== top) {
        el.style.transition = "none";
        el.style.transform = `translateY(${prev - top}px)`;
        requestAnimationFrame(() => {
          el.style.transition = "";
          el.style.transform = "";
        });
      }
      lastTops.current.set(key, top);
    });
  }, [metric]);

  return (
    <div ref={ref} className={inView ? "in" : ""}>
      <div className="board-bar">
        <div className="tabs" role="group" aria-label="Metric">
          {ORDER.map((k) => (
            <button key={k} aria-pressed={metric === k} onClick={() => setMetric(k)}>
              {METRICS[k].label}
            </button>
          ))}
        </div>
        <span className="board-note">
          {m.lowerIsBetter ? "lower is better" : "higher is better"} · {m.unit}
          {metric === "pass" && " · whisker = Wilson 95% CI"}
        </span>
      </div>

      <div className="rows">
        {rows.map((e, i) => {
          const applicable = !e.baseline || metric === "mae";
          const v = e[metric];
          const scale = applicable ? Math.min(1, v / m.max) : 0;
          return (
            <div
              key={e.model}
              ref={(el) => {
                if (el) rowRefs.current.set(e.model, el);
                else rowRefs.current.delete(e.model);
              }}
              className={`row ${e.baseline ? "baseline" : ""}`}
            >
              <span className="rank">{e.baseline ? "—" : String(i + 1).padStart(2, "0")}</span>
              <span className="name">
                {e.model}
                <small>{e.effort}</small>
              </span>
              <div className="track">
                <div
                  className="fill"
                  style={{
                    transform: `scaleX(${inView ? scale : 0})`,
                    transitionDelay: inView ? `${i * 60}ms` : undefined,
                    ["--s" as string]: scale,
                  }}
                />
                {metric === "pass" && !e.baseline && (
                  <span
                    className="ci"
                    style={{ left: `${e.passCi[0]}%`, width: `${e.passCi[1] - e.passCi[0]}%` }}
                  />
                )}
              </div>
              <span className="val">
                {applicable ? <CountUp to={v} run={inView} format={m.format} /> : "n/a"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
