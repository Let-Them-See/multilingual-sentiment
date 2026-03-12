import type { Metadata } from "next";
import Link from "next/link";
import { COLORS } from "../src/lib/constants";

export const metadata: Metadata = {
  title: "Home · Multilingual Indian Sentiment AI",
};

const BENCHMARKS = [
  { model: "IndicBERT + LoRA r=16 ⭐", f1: "85.1", lang: "6", params: "3.1M" },
  { model: "XLM-RoBERTa + LoRA r=16", f1: "82.7", lang: "6", params: "3.8M" },
  { model: "mBERT Fine-tune",          f1: "78.4", lang: "6", params: "110M" },
  { model: "XLM-RoBERTa Full FT",      f1: "81.2", lang: "6", params: "278M" },
];

const FEATURES = [
  {
    icon: "⚡",
    title: "Live Demo",
    desc: "Single & batch inference with per-language confidence breakdown.",
    href: "/demo",
    cta: "Try it",
  },
  {
    icon: "📊",
    title: "Trends Dashboard",
    desc: "30-day sentiment time-series for 10 Indian brands.",
    href: "/dashboard",
    cta: "View trends",
  },
  {
    icon: "🔬",
    title: "Ablation Research",
    desc: "8 controlled experiments: LoRA rank, data volume, loss function & more.",
    href: "/research",
    cta: "Read paper",
  },
];

export default function HomePage() {
  return (
    <div className="max-w-5xl mx-auto space-y-12">
      {/* Hero */}
      <section className="rounded-2xl px-10 py-14 text-center"
        style={{ background: `linear-gradient(135deg, ${COLORS.teal} 0%, ${COLORS.dark} 100%)` }}>
        <span className="text-5xl">🇮🇳</span>
        <h1 className="text-4xl font-bold text-white mt-4 mb-3">
          Multilingual Indian Sentiment AI
        </h1>
        <p className="text-sage text-lg max-w-2xl mx-auto mb-8">
          LoRA fine-tuned <strong style={{ color: COLORS.gold }}>IndicBERT</strong> for
          sentiment analysis across <strong style={{ color: COLORS.gold }}>6 languages</strong> —
          Hindi, Tamil, Bengali, Telugu, Marathi & code-mix. Achieves
          <strong style={{ color: COLORS.gold }}> 85.1 macro F1</strong> with only 3.1M trainable params.
        </p>
        <div className="flex flex-wrap gap-4 justify-center">
          <Link href="/demo" className="btn-primary" style={{ background: COLORS.gold, color: COLORS.dark }}>
            ⚡ Live Demo
          </Link>
          <Link href="/research" className="btn-outline" style={{ borderColor: COLORS.sage, color: COLORS.mint }}>
            📄 Read Paper
          </Link>
        </div>
      </section>

      {/* Feature Cards */}
      <section>
        <h2 className="text-2xl font-bold mb-6" style={{ color: COLORS.teal }}>
          Explore
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {FEATURES.map((f) => (
            <div key={f.title} className="card hover:shadow-card-hover transition-shadow">
              <span className="text-3xl">{f.icon}</span>
              <h3 className="text-lg font-semibold mt-3 mb-2" style={{ color: COLORS.teal }}>
                {f.title}
              </h3>
              <p className="text-sm opacity-80 mb-4">{f.desc}</p>
              <Link href={f.href} className="btn-primary text-sm">
                {f.cta} →
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* Benchmark Table */}
      <section>
        <h2 className="text-2xl font-bold mb-6" style={{ color: COLORS.teal }}>
          Benchmark Results
        </h2>
        <div className="rounded-xl overflow-hidden shadow-card">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: COLORS.teal, color: COLORS.mint }}>
                <th className="px-5 py-3 text-left font-semibold">Model</th>
                <th className="px-5 py-3 text-center font-semibold">Macro F1</th>
                <th className="px-5 py-3 text-center font-semibold">Languages</th>
                <th className="px-5 py-3 text-center font-semibold">Trainable Params</th>
              </tr>
            </thead>
            <tbody>
              {BENCHMARKS.map((row, i) => (
                <tr
                  key={row.model}
                  style={{
                    background: row.model.includes("⭐")
                      ? COLORS.gold
                      : i % 2 === 0
                      ? COLORS.sage
                      : COLORS.mint,
                    color: COLORS.dark,
                    fontWeight: row.model.includes("⭐") ? 700 : 400,
                  }}
                >
                  <td className="px-5 py-3">{row.model}</td>
                  <td className="px-5 py-3 text-center">{row.f1}</td>
                  <td className="px-5 py-3 text-center">{row.lang}</td>
                  <td className="px-5 py-3 text-center">{row.params}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Tech Stack */}
      <section>
        <h2 className="text-2xl font-bold mb-6" style={{ color: COLORS.teal }}>
          Tech Stack
        </h2>
        <div className="flex flex-wrap gap-3">
          {[
            "HuggingFace Transformers", "PEFT / LoRA", "BitsAndBytes 4-bit",
            "Weights & Biases", "FastAPI", "Next.js 14", "Gradio",
            "Redis Cache", "Docker", "IndicBERT",
          ].map((tag) => (
            <span
              key={tag}
              className="text-xs font-medium px-3 py-1.5 rounded-full"
              style={{ background: COLORS.sage, color: COLORS.teal }}
            >
              {tag}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}
