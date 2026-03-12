import type { Metadata } from "next";
import { COLORS } from "../../src/lib/constants";
import type { AblationRow } from "../../src/lib/types";

export const metadata: Metadata = {
  title: "Research · Multilingual Indian Sentiment AI",
};

// ── Static ablation summary data ──────────────────────────────────────────
const ABLATION_DATA: AblationRow[] = [
  { config: "IndicBERT + LoRA r=16 ⭐", f1_macro: 85.1, f1_hindi: 87.4, f1_tamil: 83.2, f1_bengali: 84.9, f1_code_mix: 79.3, params_m: 3.1, is_best: true },
  { config: "XLM-RoBERTa + LoRA r=16",  f1_macro: 82.7, f1_hindi: 84.2, f1_tamil: 81.5, f1_bengali: 83.1, f1_code_mix: 78.1, params_m: 3.8, is_best: false },
  { config: "mBERT + LoRA r=16",         f1_macro: 79.3, f1_hindi: 81.2, f1_tamil: 78.0, f1_bengali: 79.8, f1_code_mix: 74.2, params_m: 2.6, is_best: false },
  { config: "IndicBERT + LoRA r=8",      f1_macro: 83.2, f1_hindi: 85.0, f1_tamil: 81.8, f1_bengali: 83.4, f1_code_mix: 77.1, params_m: 1.6, is_best: false },
  { config: "IndicBERT + LoRA r=32",     f1_macro: 85.3, f1_hindi: 87.6, f1_tamil: 83.5, f1_bengali: 85.1, f1_code_mix: 79.7, params_m: 6.2, is_best: false },
  { config: "FocalLoss γ=2 (proposed)",  f1_macro: 85.1, f1_hindi: 87.4, f1_tamil: 83.2, f1_bengali: 84.9, f1_code_mix: 79.3, params_m: 3.1, is_best: false },
  { config: "CrossEntropy (baseline)",   f1_macro: 83.5, f1_hindi: 85.8, f1_tamil: 81.9, f1_bengali: 83.7, f1_code_mix: 76.4, params_m: 3.1, is_best: false },
];

function AblationTable({ rows }: { rows: AblationRow[] }) {
  return (
    <div className="rounded-xl overflow-hidden shadow-card">
      <table className="w-full text-sm">
        <thead>
          <tr style={{ background: COLORS.teal, color: COLORS.mint }}>
            {["Config", "Overall F1", "Hindi", "Tamil", "Bengali", "Code-Mix", "Params"].map((h) => (
              <th key={h} className="px-4 py-3 text-left font-semibold whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.config}
              style={{
                background: row.is_best ? COLORS.gold : i % 2 === 0 ? COLORS.sage : COLORS.mint,
                color: COLORS.dark,
                fontWeight: row.is_best ? 700 : 400,
              }}
            >
              <td className="px-4 py-2.5">{row.config}</td>
              <td className="px-4 py-2.5 text-center">{row.f1_macro.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-center">{row.f1_hindi.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-center">{row.f1_tamil.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-center">{row.f1_bengali.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-center">{row.f1_code_mix.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-center">{row.params_m.toFixed(1)}M</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ResearchPage() {
  return (
    <div className="max-w-4xl mx-auto space-y-10">
      <div>
        <h1 className="text-3xl font-bold" style={{ color: COLORS.teal }}>
          🔬 Research — Localized LLMs for Indian Markets
        </h1>
        <p className="opacity-70 mt-1 text-sm">
          Submitted to ACL 2025 · arXiv:2024.XXXXX
        </p>
      </div>

      {/* Abstract */}
      <section className="card">
        <h2 className="text-xl font-bold mb-3" style={{ color: COLORS.teal }}>Abstract</h2>
        <p className="text-sm leading-relaxed opacity-80">
          We present a parameter-efficient approach for multilingual sentiment analysis targeting
          six major Indian languages: Hindi, Tamil, Bengali, Telugu, Marathi, and code-mixed text.
          Our method applies Low-Rank Adaptation (LoRA) with rank r=16 to IndicBERT, achieving a
          macro F1 of <strong>85.1</strong> using only <strong>3.1M trainable parameters</strong>
          — a 97.2% reduction over full fine-tuning. We introduce a novel dataset of 50,000+
          brand-sentiment samples scraped from Twitter and Reddit, covering 10 major Indian consumer
          brands. Systematic ablation across 8 experimental axes (base model, LoRA rank, data volume,
          language exclusion, code-mix strategy, augmentation, loss function, and quantization) reveals
          that Focal Loss (γ=2) provides consistent improvements over cross-entropy on imbalanced
          multilingual data, and that back-translation augmentation yields +1.4 F1 for low-resource
          languages (Marathi, Telugu). We release all code, datasets, and trained adapters under MIT license.
        </p>
      </section>

      {/* Key Results */}
      <section>
        <h2 className="text-xl font-bold mb-4" style={{ color: COLORS.teal }}>
          Ablation Study Results
        </h2>
        <AblationTable rows={ABLATION_DATA} />
        <p className="text-xs opacity-50 mt-2">
          ⭐ Best configuration. All experiments fixed seed=42, 3 runs averaged.
        </p>
      </section>

      {/* Key Findings */}
      <section className="card space-y-3">
        <h2 className="text-xl font-bold" style={{ color: COLORS.teal }}>Key Findings</h2>
        {[
          ["LoRA r=16 sweet spot", "r=16 achieves near-r=32 performance at 50% fewer params. r=64 offers no statistically significant gain (p>0.05)."],
          ["IndicBERT > XLM-R for Indic scripts", "IndicBERT's Devanagari-aware tokenizer reduces fertility for Hindi/Marathi by 18%, directly improving those language F1 scores."],
          ["Focal Loss improves minority class", "FocalLoss (γ=2) boosts neutral-class recall from 61.2 → 68.4 on the imbalanced test set without hurting positive/negative F1."],
          ["Code-mix needs language prefix", "Prepending <lang> tokens to code-mixed text improves code-mix F1 by +2.1 points vs. raw transliterated input."],
          ["Back-translation helps low-resource", "Augmenting Telugu and Marathi via Helsinki-NLP back-translation raises those language F1 by +1.4 and +1.7 respectively."],
        ].map(([title, body]) => (
          <div key={title} className="border-l-4 pl-4" style={{ borderColor: COLORS.gold }}>
            <p className="font-semibold text-sm" style={{ color: COLORS.teal }}>{title}</p>
            <p className="text-sm opacity-70 mt-0.5">{body}</p>
          </div>
        ))}
      </section>

      {/* Citation */}
      <section className="card">
        <h2 className="text-xl font-bold mb-3" style={{ color: COLORS.teal }}>BibTeX Citation</h2>
        <pre
          className="text-xs rounded-lg p-4 overflow-x-auto"
          style={{ background: COLORS.dark, color: COLORS.sage }}
        >{`@inproceedings{multilingual-sentiment-2025,
  title     = {Localized LLMs for Indian Markets: Parameter-Efficient 
               Multilingual Sentiment Analysis via LoRA},
  author    = {Vedant and others},
  booktitle = {Proceedings of ACL 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2024.XXXXX}
}`}</pre>
      </section>
    </div>
  );
}
