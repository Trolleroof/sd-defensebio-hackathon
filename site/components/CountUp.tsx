"use client";

import { useEffect, useState } from "react";

export function CountUp({
  to,
  run,
  format,
  ms = 1100,
}: {
  to: number;
  run: boolean;
  format: (v: number) => string;
  ms?: number;
}) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!run) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setValue(to);
      return;
    }
    let raf = 0;
    const from = value;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(from + (to - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // Animate from the current value whenever the target changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [to, run, ms]);

  return <>{format(value)}</>;
}
