"use client";

import { useEffect, useState } from "react";
import { POST_SECTIONS } from "@/lib/data";

export function TocSpy() {
  const [active, setActive] = useState(POST_SECTIONS[0].id);

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length) setActive(visible[0].target.id);
      },
      { rootMargin: "-20% 0px -70% 0px" },
    );
    POST_SECTIONS.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, []);

  return (
    <aside>
      {POST_SECTIONS.map((s) => (
        <a key={s.id} href={`#${s.id}`} className={active === s.id ? "active" : ""}>
          <span>{s.n}</span>
          {s.title}
        </a>
      ))}
    </aside>
  );
}
