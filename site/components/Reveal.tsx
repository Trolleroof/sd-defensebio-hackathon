"use client";

import type { ElementType, ReactNode } from "react";
import { useInView } from "@/lib/useInView";

export function Reveal({
  as: Tag = "div",
  stagger = false,
  className = "",
  children,
  ...rest
}: {
  as?: ElementType;
  stagger?: boolean;
  className?: string;
  children: ReactNode;
  [key: string]: unknown;
}) {
  const [ref, inView] = useInView<HTMLElement>(0.12);
  const base = stagger ? "stagger" : "reveal";
  return (
    <Tag ref={ref} className={`${base} ${inView ? "in" : ""} ${className}`} {...rest}>
      {children}
    </Tag>
  );
}
