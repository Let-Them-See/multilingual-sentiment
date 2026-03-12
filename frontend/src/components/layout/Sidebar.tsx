"use client";

import Link from "next/link";
import { NAV_LINKS, COLORS } from "../../lib/constants";
import { usePathname } from "next/navigation";

// ── Stats ──────────────────────────────────────────────────────────────────
const STATS = [
  { label: "Languages", value: "6" },
  { label: "Brands", value: "10" },
  { label: "Samples", value: "50k+" },
  { label: "F1 Macro", value: "85.1" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="fixed left-0 top-0 h-screen w-64 flex flex-col"
      style={{ background: COLORS.dark }}
    >
      {/* Logo */}
      <div className="px-6 py-6 border-b border-teal/30">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🇮🇳</span>
          <div>
            <p className="text-mint font-bold text-sm leading-tight">
              Multilingual
            </p>
            <p style={{ color: COLORS.gold }} className="font-bold text-sm leading-tight">
              Sentiment AI
            </p>
          </div>
        </div>
        <p className="text-sage text-xs mt-2 opacity-70">
          LoRA fine-tuned · 6 Indian languages
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        {NAV_LINKS.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`nav-link ${isActive ? "nav-link-active" : ""}`}
              style={isActive ? { background: COLORS.teal } : {}}
            >
              <span className="text-base">{link.icon}</span>
              <span>{link.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Quick Stats */}
      <div className="px-4 py-4 border-t border-teal/30">
        <p className="text-xs text-sage/60 uppercase tracking-widest mb-3">
          Model Stats
        </p>
        <div className="grid grid-cols-2 gap-2">
          {STATS.map((s) => (
            <div
              key={s.label}
              className="rounded-lg p-2 text-center"
              style={{ background: COLORS.teal }}
            >
              <p style={{ color: COLORS.gold }} className="font-bold text-sm">
                {s.value}
              </p>
              <p className="text-sage text-xs">{s.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-teal/30">
        <p className="text-sage/50 text-xs text-center">
          v1.0 · ACL 2025 Submission
        </p>
      </div>
    </aside>
  );
}
