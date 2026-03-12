// ── Design System Constants ────────────────────────────────────────────────
export const COLORS = {
  mint:   "#F1F6F4",
  gold:   "#FFC801",
  teal:   "#114C5A",
  sage:   "#D9E8E2",
  orange: "#FF9932",
  dark:   "#172B36",
} as const;

// Recharts-friendly sentiment color map
export const SENTIMENT_COLORS: Record<string, string> = {
  positive: COLORS.teal,
  neutral:  COLORS.gold,
  negative: COLORS.orange,
};

// ── API Config ─────────────────────────────────────────────────────────────
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const API_V1 = `${API_BASE}/api/v1`;

// ── Supported Languages ────────────────────────────────────────────────────
export const LANGUAGES: Record<string, string> = {
  hi:       "Hindi",
  ta:       "Tamil",
  bn:       "Bengali",
  te:       "Telugu",
  mr:       "Marathi",
  code_mix: "Code-Mix",
};

// ── Brands ─────────────────────────────────────────────────────────────────
export const BRANDS = [
  "Jio", "Airtel", "Zomato", "Swiggy", "Flipkart",
  "Amazon", "Meesho", "Nykaa", "LensKart", "BigBasket",
] as const;

// ── Nav Links ──────────────────────────────────────────────────────────────
export const NAV_LINKS = [
  { href: "/",          label: "Home",       icon: "🏠" },
  { href: "/demo",      label: "Live Demo",  icon: "⚡" },
  { href: "/dashboard", label: "Dashboard",  icon: "📊" },
  { href: "/research",  label: "Research",   icon: "🔬" },
] as const;
