"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { DASHBOARD_URL } from "@/lib/data";

const LINKS = [
  { href: "/", label: "Leaderboard" },
  { href: "/blog/physbench/", label: "Report" },
  { href: "/tasks/", label: "Tasks" },
  { href: "/traces/", label: "Judge traces" },
  { href: "/dashboard/", label: "Dashboard" },
];

export function Header() {
  const path = usePathname();
  return (
    <header className="site-header">
      <div className="wrap">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden="true" />
          soliderphys-bench
        </Link>
        <nav className="nav">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href.replace(/\/$/, ""));
            return (
              <Link key={l.href} href={l.href} aria-current={active ? "page" : undefined}>
                {l.label}
              </Link>
            );
          })}
        </nav>
        <a className="header-cta" href={DASHBOARD_URL}>
          Open dashboard ↗
        </a>
      </div>
    </header>
  );
}
