import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { Header } from "@/components/Header";
import { DASHBOARD_URL } from "@/lib/data";
import "./globals.css";

const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-sans" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "PhysBench",
  description: "Benchmark for agents that build personalized heart-rate predictors.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        {/* Scroll reveals only hide content once JS is known to be running. */}
        <script dangerouslySetInnerHTML={{ __html: "document.documentElement.classList.add('js')" }} />
      </head>
      <body>
        <Header />
        <main>{children}</main>
        <footer className="site-footer">
          <div className="wrap">
            <span>PhysBench · DefenseBio</span>
            <span style={{ display: "flex", gap: 20 }}>
              <a href="/blog/physbench/">Report</a>
              <a href="/tasks/">Tasks</a>
              <a href={DASHBOARD_URL}>Operator dashboard ↗</a>
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
