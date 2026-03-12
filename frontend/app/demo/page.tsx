"use client";

import { useState } from "react";
import { predictSingle, predictBatch } from "../../src/lib/api";
import type { PredictResponse } from "../../src/lib/types";
import { COLORS, LANGUAGES } from "../../src/lib/constants";

const LANG_OPTIONS = [
  { value: "", label: "Auto-detect" },
  ...Object.entries(LANGUAGES).map(([v, l]) => ({ value: v, label: l })),
];

const DEMO_TEXTS = [
  "Jio का नेटवर्क बहुत अच्छा है, signal हमेशा strong रहता है! 😊",
  "Zomato delivery yesterday was so late, food was cold and pathetic.",
  "Flipkart சேவை நல்லது, விரைவாக deliver செய்தார்கள்.",
  "Amazon er service khub valo, product genuine chilo.",
  "Swiggy chi service khup chaan ahe, delivery fast hote.",
  "Meesho ka product ekdum bakwaas hai yaar, waste of money",
];

function SentimentBadge({ label }: { label: string }) {
  const style: Record<string, React.CSSProperties> = {
    positive: { background: COLORS.teal, color: COLORS.mint },
    neutral:  { background: COLORS.gold, color: COLORS.dark },
    negative: { background: COLORS.orange, color: "#fff" },
  };
  return (
    <span
      className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-bold"
      style={style[label] ?? style.neutral}
    >
      {label === "positive" ? "😊" : label === "negative" ? "😞" : "😐"}{" "}
      {label.charAt(0).toUpperCase() + label.slice(1)}
    </span>
  );
}

function ConfidenceBar({ label, value }: { label: string; value: number }) {
  const colors: Record<string, string> = {
    positive: COLORS.teal,
    neutral: COLORS.gold,
    negative: COLORS.orange,
  };
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-medium">
        <span style={{ color: COLORS.teal }} className="capitalize">{label}</span>
        <span style={{ color: COLORS.dark }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full" style={{ background: COLORS.sage }}>
        <div
          className="h-2 rounded-full transition-all duration-500"
          style={{ width: `${value * 100}%`, background: colors[label] ?? COLORS.teal }}
        />
      </div>
    </div>
  );
}

export default function DemoPage() {
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictSingle(
        text.trim(),
        language || undefined,
      );
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDemo = (sample: string) => {
    setText(sample);
    setResult(null);
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold" style={{ color: COLORS.teal }}>
          ⚡ Live Sentiment Demo
        </h1>
        <p className="opacity-70 mt-1">
          Classify Hindi, Tamil, Bengali, Telugu, Marathi, or code-mix text in real time.
        </p>
      </div>

      {/* Input Card */}
      <div className="card space-y-4">
        <div>
          <label className="block text-sm font-semibold mb-1.5" style={{ color: COLORS.teal }}>
            Input Text
          </label>
          <textarea
            className="input-field min-h-[100px] resize-none"
            placeholder="Type or paste any Indian language text…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            maxLength={512}
          />
          <p className="text-xs opacity-50 mt-1 text-right">{text.length}/512</p>
        </div>

        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm font-semibold mb-1.5" style={{ color: COLORS.teal }}>
              Language (optional)
            </label>
            <select
              className="input-field"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              {LANG_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          <button
            className="btn-primary"
            onClick={handlePredict}
            disabled={loading || !text.trim()}
            style={{ background: COLORS.gold, color: COLORS.dark }}
          >
            {loading ? "Analysing…" : "Analyse →"}
          </button>
        </div>

        {error && (
          <div
            className="rounded-lg px-4 py-3 text-sm font-medium"
            style={{ background: "#FFF3CD", color: COLORS.orange }}
          >
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Result */}
      {result && (
        <div className="card space-y-5">
          <div className="flex items-center justify-between">
            <SentimentBadge label={result.prediction.label} />
            <span className="text-xs opacity-50">
              {result.prediction.inference_ms.toFixed(1)} ms · {result.model_version}
            </span>
          </div>

          <div className="space-y-3">
            {Object.entries(result.prediction.probabilities).map(([lbl, prob]) => (
              <ConfidenceBar key={lbl} label={lbl} value={prob} />
            ))}
          </div>

          {result.prediction.language_detected && (
            <p className="text-xs opacity-50">
              Detected language:{" "}
              <strong>{result.prediction.language_detected}</strong>
            </p>
          )}
        </div>
      )}

      {/* Sample Texts */}
      <div>
        <h3 className="text-sm font-semibold mb-3" style={{ color: COLORS.teal }}>
          Try sample texts:
        </h3>
        <div className="space-y-2">
          {DEMO_TEXTS.map((sample, i) => (
            <button
              key={i}
              onClick={() => handleDemo(sample)}
              className="w-full text-left text-sm px-4 py-3 rounded-lg transition-colors"
              style={{
                background: i % 2 === 0 ? COLORS.sage : COLORS.mint,
                color: COLORS.dark,
                border: `1px solid ${COLORS.sage}`,
              }}
            >
              {sample}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
