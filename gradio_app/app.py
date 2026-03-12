"""
gradio_app/app.py
Multilingual Indian Sentiment Analysis — Gradio Demo
4-tab interface: Single Inference | Batch CSV | Brand Trends | Bias Audit
Palette: mint=#F1F6F4, gold=#FFC801, teal=#114C5A, sage=#D9E8E2, orange=#FF9932, dark=#172B36
"""

import os
import io
import csv
import json
import random
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import gradio as gr

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

COLORS = {
    "mint":   "#F1F6F4",
    "gold":   "#FFC801",
    "teal":   "#114C5A",
    "sage":   "#D9E8E2",
    "orange": "#FF9932",
    "dark":   "#172B36",
}

LANGUAGES = {
    "auto": "Auto-detect",
    "hi":   "Hindi (हिन्दी)",
    "ta":   "Tamil (தமிழ்)",
    "bn":   "Bengali (বাংলা)",
    "mr":   "Marathi (मराठी)",
    "te":   "Telugu (తెలుగు)",
    "en":   "English",
}

BRANDS = [
    "Jio", "Flipkart", "Zomato", "Amazon India",
    "Swiggy", "Ola", "Paytm", "BYJU's", "CRED", "PhonePe",
]

SAMPLE_TEXTS = {
    "Hindi":   "जियो का नेटवर्क अब पहले से बेहतर हो गया है, सच में बहुत तेज़ है!",
    "Tamil":   "ஜோமேட்டோ டெலிவரி மிகவும் தாமதமாக வந்தது, மிகவும் ஏமாற்றமாக உள்ளது.",
    "Bengali": "ফ্লিপকার্টের ডেলিভারি সার্ভিস আজকাল অনেক ভালো হয়েছে।",
    "Marathi": "पेटीएम वापरणे खूप सोपे आहे आणि पैसे पाठवणे झटपट होते.",
    "Telugu":  "ओला యాప్ చాలా బాగుంది, డ్రైవర్ కూడా మర్యాద గా ఉన్నారు.",
    "English": "Amazon India's same-day delivery is absolutely fantastic — couldn't be happier!",
}

# ──────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────
CUSTOM_CSS = f"""
:root {{
  --mint:   {COLORS['mint']};
  --gold:   {COLORS['gold']};
  --teal:   {COLORS['teal']};
  --sage:   {COLORS['sage']};
  --orange: {COLORS['orange']};
  --dark:   {COLORS['dark']};
}}

body, .gradio-container {{
  background-color: var(--mint) !important;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  color: var(--dark) !important;
}}

h1, h2, h3 {{
  color: var(--teal) !important;
}}

.gr-button-primary {{
  background-color: var(--teal) !important;
  border-color: var(--teal) !important;
  color: var(--mint) !important;
}}

.gr-button-secondary {{
  background-color: transparent !important;
  border-color: var(--teal) !important;
  color: var(--teal) !important;
}}

.gr-button-primary:hover {{
  background-color: var(--dark) !important;
  border-color: var(--dark) !important;
}}

.gr-form, .gr-box {{
  background-color: var(--sage) !important;
  border-color: #c5d9d0 !important;
  border-radius: 10px !important;
}}

.gr-input, .gr-textarea, .gr-dropdown {{
  background-color: white !important;
  border-color: var(--sage) !important;
  border-radius: 8px !important;
  color: var(--dark) !important;
}}

.gr-tab-nav button {{
  color: var(--dark) !important;
  border-bottom: 2px solid transparent;
}}

.gr-tab-nav button.selected {{
  color: var(--teal) !important;
  border-bottom: 2px solid var(--teal) !important;
  font-weight: 600;
}}

.label-positive  {{ color: {COLORS['teal']}   !important; font-weight: 700; }}
.label-neutral   {{ color: {COLORS['gold']}   !important; font-weight: 700; }}
.label-negative  {{ color: {COLORS['orange']} !important; font-weight: 700; }}

.metric-badge {{
  display: inline-block;
  padding: 4px 12px;
  border-radius: 9999px;
  font-weight: 600;
  font-size: 0.85rem;
}}

footer {{ display: none !important; }}
"""

# ──────────────────────────────────────────────
#  API Helpers
# ──────────────────────────────────────────────

def _api_predict_single(text: str, language: str) -> dict:
    payload = {"text": text}
    if language != "auto":
        payload["language"] = language
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _api_predict_batch(texts: list[str]) -> dict:
    try:
        r = requests.post(f"{API_BASE}/predict/batch", json={"texts": texts}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _api_trends(brand: str, days: int) -> dict:
    try:
        r = requests.get(f"{API_BASE}/trends", params={"brand": brand, "days": days}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _api_bias(texts: list[str], labels: list[int]) -> dict:
    try:
        r = requests.post(f"{API_BASE}/bias/check",
                          json={"texts": texts, "true_labels": labels}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────
#  Mock fallback (when backend is offline)
# ──────────────────────────────────────────────

def _mock_predict(text: str) -> dict:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000
    rng = random.Random(seed)
    labels = ["positive", "neutral", "negative"]
    probs = sorted([rng.random() for _ in range(3)], reverse=True)
    total = sum(probs)
    probs = [p / total for p in probs]
    label = labels[0]
    return {
        "label": label,
        "label_id": 0,
        "confidence": round(probs[0], 4),
        "probabilities": {"positive": round(probs[0], 4),
                          "neutral":  round(probs[1], 4),
                          "negative": round(probs[2], 4)},
        "inference_ms": round(rng.uniform(12, 45), 2),
        "_mock": True,
    }


def _mock_trends(brand: str, days: int) -> list[dict]:
    rng = random.Random(hash(brand))
    pos, neu, neg = 0.5, 0.3, 0.2
    rows = []
    base_date = datetime.now() - timedelta(days=days)
    for i in range(days):
        pos += rng.uniform(-0.02, 0.02)
        neu += rng.uniform(-0.01, 0.01)
        pos = max(0.1, min(0.8, pos))
        neu = max(0.1, min(0.5, neu))
        neg = max(0.0, 1.0 - pos - neu)
        rows.append({
            "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "positive": round(pos, 3),
            "neutral":  round(neu, 3),
            "negative": round(neg, 3),
        })
    return rows


# ──────────────────────────────────────────────
#  Tab 1 — Single Inference
# ──────────────────────────────────────────────

def run_single_inference(text: str, language: str) -> tuple[str, str, str]:
    """Returns (sentiment_html, confidence_html, stats_html)."""
    if not text.strip():
        return "<p style='color:gray'>Please enter some text.</p>", "", ""

    lang_code = language.split(" ")[0].lower() if language != "Auto-detect" else "auto"
    data = _api_predict_single(text, lang_code)

    if "error" in data:
        data = _mock_predict(text)
        mock_note = " <small style='color:gray'>(mock — backend offline)</small>"
    else:
        mock_note = ""

    label = data.get("label", "unknown")
    confidence = data.get("confidence", 0.0)
    probs = data.get("probabilities", {})
    inf_ms = data.get("inference_ms", 0)

    color_map = {"positive": COLORS["teal"], "neutral": COLORS["gold"], "negative": COLORS["orange"]}
    badge_color = color_map.get(label, COLORS["dark"])

    sentiment_html = f"""
    <div style='text-align:center;padding:24px;background:{COLORS["sage"]};border-radius:12px'>
      <div style='font-size:2.5rem;margin-bottom:8px'>
        {"😊" if label=="positive" else "😐" if label=="neutral" else "😞"}
      </div>
      <div style='display:inline-block;padding:6px 20px;background:{badge_color};
                  color:white;border-radius:9999px;font-size:1.1rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.05em'>
        {label}{mock_note}
      </div>
      <p style='margin-top:12px;color:{COLORS["dark"]};font-size:0.9rem'>
        Confidence: <strong style='color:{badge_color}'>{confidence*100:.1f}%</strong>
        &nbsp;·&nbsp; Inference: <strong>{inf_ms:.1f} ms</strong>
      </p>
    </div>
    """

    bars = ""
    for lbl, prob in probs.items():
        c = color_map.get(lbl, COLORS["dark"])
        bars += f"""
        <div style='margin-bottom:10px'>
          <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
            <span style='font-weight:600;text-transform:capitalize;color:{COLORS["dark"]}'>{lbl}</span>
            <span style='color:{c};font-weight:700'>{prob*100:.1f}%</span>
          </div>
          <div style='background:#e2e8f0;border-radius:9999px;height:10px;overflow:hidden'>
            <div style='width:{prob*100:.1f}%;background:{c};height:100%;
                        border-radius:9999px;transition:width 0.5s ease'></div>
          </div>
        </div>"""

    confidence_html = f"""
    <div style='padding:16px;background:white;border-radius:10px;border:1px solid {COLORS["sage"]}'>
      <h4 style='margin:0 0 14px;color:{COLORS["teal"]}'>Probability Distribution</h4>
      {bars}
    </div>
    """

    stats_html = f"""
    <div style='padding:12px 16px;background:{COLORS["dark"]};border-radius:8px;font-family:monospace'>
      <span style='color:{COLORS["sage"]}'># Model Output</span><br/>
      <span style='color:{COLORS["gold"]}'>label</span>
      <span style='color:white'> = "{label}"</span><br/>
      <span style='color:{COLORS["gold"]}'>confidence</span>
      <span style='color:white'> = {confidence:.4f}</span><br/>
      <span style='color:{COLORS["gold"]}'>inference_ms</span>
      <span style='color:white'> = {inf_ms}</span>
    </div>
    """

    return sentiment_html, confidence_html, stats_html


def load_sample(sample_name: str) -> str:
    return SAMPLE_TEXTS.get(sample_name, "")


# ──────────────────────────────────────────────
#  Tab 2 — Batch CSV
# ──────────────────────────────────────────────

def run_batch_inference(file_obj) -> tuple[str, str]:
    """Returns (results_html, download_csv_path)."""
    if file_obj is None:
        return "<p style='color:gray'>Please upload a CSV file.</p>", None

    try:
        content = file_obj.decode("utf-8") if isinstance(file_obj, bytes) else file_obj.read()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
    except Exception as e:
        return f"<p style='color:red'>CSV parse error: {e}</p>", None

    if not rows:
        return "<p style='color:gray'>CSV is empty.</p>", None

    text_col = next((c for c in rows[0] if "text" in c.lower()), None)
    if not text_col:
        text_col = list(rows[0].keys())[0]

    texts = [r[text_col] for r in rows[:50]]
    data = _api_predict_batch(texts)

    if "error" in data:
        results = [_mock_predict(t) for t in texts]
        mock_note = " (mock)"
    else:
        results = data.get("results", [])
        mock_note = ""

    color_map = {"positive": COLORS["teal"], "neutral": COLORS["gold"], "negative": COLORS["orange"]}

    table_rows = ""
    for i, (text, res) in enumerate(zip(texts, results)):
        label = res.get("label", "?")
        conf  = res.get("confidence", 0)
        c = color_map.get(label, "#666")
        bg = COLORS["sage"] if i % 2 == 0 else "white"
        table_rows += f"""
        <tr style='background:{bg}'>
          <td style='padding:8px 12px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{text[:80]}</td>
          <td style='padding:8px 12px'>
            <span style='background:{c};color:white;padding:2px 10px;border-radius:9999px;
                         font-size:0.8rem;font-weight:600;text-transform:uppercase'>{label}</span>
          </td>
          <td style='padding:8px 12px;color:{c};font-weight:700'>{conf*100:.1f}%</td>
        </tr>"""

    results_html = f"""
    <div style='overflow-x:auto'>
      <p style='color:{COLORS["teal"]};font-weight:600;margin-bottom:8px'>
        Processed {len(texts)} texts{mock_note}
      </p>
      <table style='width:100%;border-collapse:collapse;font-size:0.9rem'>
        <thead>
          <tr style='background:{COLORS["teal"]};color:white'>
            <th style='padding:10px 12px;text-align:left'>Text</th>
            <th style='padding:10px 12px;text-align:left'>Sentiment</th>
            <th style='padding:10px 12px;text-align:left'>Confidence</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    """

    # Write downloadable CSV
    out_path = "/tmp/batch_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "sentiment", "confidence"])
        for text, res in zip(texts, results):
            writer.writerow([text, res.get("label", "?"), res.get("confidence", 0)])

    return results_html, out_path


# ──────────────────────────────────────────────
#  Tab 3 — Brand Trends
# ──────────────────────────────────────────────

def load_brand_trends(brand: str, days: int) -> str:
    data = _api_trends(brand, days)
    if "error" in data:
        trend_points = _mock_trends(brand, days)
        note = " (mock data — backend offline)"
    else:
        trend_points = data.get("trend", [])
        note = ""

    if not trend_points:
        return "<p>No trend data available.</p>"

    # Build simple inline SVG bar chart
    dates = [p["date"] for p in trend_points]
    pos_vals = [p["positive"] for p in trend_points]
    neg_vals = [p["negative"] for p in trend_points]
    neu_vals = [p["neutral"]  for p in trend_points]

    n = len(dates)
    w = max(600, n * 8)
    h = 200
    bar_w = max(4, w // n - 1)

    bars_svg = ""
    for i, (p, neu, ng) in enumerate(zip(pos_vals, neu_vals, neg_vals)):
        x = i * (bar_w + 1)
        p_h  = int(p   * h)
        n_h  = int(neu * h)
        ng_h = int(ng  * h)
        y2 = h - p_h
        y1 = y2 - n_h
        y0 = y1 - ng_h
        bars_svg += (
            f'<rect x="{x}" y="{y2}" width="{bar_w}" height="{p_h}" fill="{COLORS["teal"]}" opacity="0.85"/>'
            f'<rect x="{x}" y="{y1}" width="{bar_w}" height="{n_h}" fill="{COLORS["gold"]}" opacity="0.85"/>'
            f'<rect x="{x}" y="{y0}" width="{bar_w}" height="{ng_h}" fill="{COLORS["orange"]}" opacity="0.85"/>'
        )

    avg_pos = sum(pos_vals) / len(pos_vals)
    avg_neg = sum(neg_vals) / len(neg_vals)

    html = f"""
    <div style='padding:16px;background:white;border-radius:10px;border:1px solid {COLORS["sage"]}'>
      <h4 style='margin:0 0 4px;color:{COLORS["teal"]}'>{brand} — Sentiment Trend ({days}d){note}</h4>
      <div style='display:flex;gap:16px;margin-bottom:12px;font-size:0.85rem'>
        <span style='display:flex;align-items:center;gap:6px'>
          <span style='width:12px;height:12px;background:{COLORS["teal"]};border-radius:2px;display:inline-block'></span>Positive
        </span>
        <span style='display:flex;align-items:center;gap:6px'>
          <span style='width:12px;height:12px;background:{COLORS["gold"]};border-radius:2px;display:inline-block'></span>Neutral
        </span>
        <span style='display:flex;align-items:center;gap:6px'>
          <span style='width:12px;height:12px;background:{COLORS["orange"]};border-radius:2px;display:inline-block'></span>Negative
        </span>
      </div>
      <div style='overflow-x:auto'>
        <svg viewBox="0 0 {w} {h}" style="width:100%;height:200px">
          {bars_svg}
        </svg>
      </div>
      <div style='display:flex;gap:24px;margin-top:12px;font-size:0.9rem'>
        <div style='background:{COLORS["sage"]};padding:10px 16px;border-radius:8px'>
          <div style='color:{COLORS["teal"]};font-weight:700;font-size:1.2rem'>{avg_pos*100:.1f}%</div>
          <div style='color:{COLORS["dark"]}'>Avg Positive</div>
        </div>
        <div style='background:{COLORS["sage"]};padding:10px 16px;border-radius:8px'>
          <div style='color:{COLORS["orange"]};font-weight:700;font-size:1.2rem'>{avg_neg*100:.1f}%</div>
          <div style='color:{COLORS["dark"]}'>Avg Negative</div>
        </div>
        <div style='background:{COLORS["sage"]};padding:10px 16px;border-radius:8px'>
          <div style='color:{COLORS["dark"]};font-weight:700;font-size:1.2rem'>{dates[0]} → {dates[-1]}</div>
          <div style='color:{COLORS["dark"]}'>Date Range</div>
        </div>
      </div>
    </div>
    """
    return html


# ──────────────────────────────────────────────
#  Tab 4 — Bias Audit
# ──────────────────────────────────────────────

def run_bias_audit(file_obj) -> str:
    if file_obj is None:
        return "<p style='color:gray'>Upload a labeled CSV with columns: text, label (0=pos, 1=neu, 2=neg)</p>"

    try:
        content = file_obj.decode("utf-8") if isinstance(file_obj, bytes) else file_obj.read()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
    except Exception as e:
        return f"<p style='color:red'>CSV error: {e}</p>"

    texts  = [r.get("text", "") for r in rows[:200]]
    labels = []
    for r in rows[:200]:
        try:
            labels.append(int(r.get("label", r.get("sentiment", 0))))
        except ValueError:
            labels.append(0)

    data = _api_bias(texts, labels)

    if "error" in data:
        # Simple mock bias report
        return f"""
        <div style='padding:16px;background:{COLORS["sage"]};border-radius:10px'>
          <h4 style='color:{COLORS["teal"]}'>Bias Audit (Mock — Backend Offline)</h4>
          <p style='color:{COLORS["dark"]}'>Backend connection failed. In production, this would show:</p>
          <ul style='color:{COLORS["dark"]}'>
            <li>Gender bias gap (m vs f templates)</li>
            <li>Regional bias gap (metro vs tier-2)</li>
            <li>Script bias gap (Devanagari vs Latin)</li>
            <li>Brand bias gap (major vs niche)</li>
            <li>Class imbalance metrics</li>
          </ul>
          <p style='color:{COLORS["orange"]};font-style:italic'>Start the FastAPI backend: <code>uvicorn backend.main:app</code></p>
        </div>
        """

    overall = data.get("overall_bias_score", 0.0)
    flags   = data.get("bias_flags", [])
    dims    = data.get("dimensions", {})

    flag_color = COLORS["orange"] if flags else COLORS["teal"]
    flag_icon  = "⚠️" if flags else "✅"

    dim_cards = ""
    for dim_name, dim_data in dims.items():
        gap = dim_data.get("gap", 0)
        bar_pct = min(100, abs(gap) * 1000)
        bar_color = COLORS["orange"] if abs(gap) > 0.05 else COLORS["teal"]
        dim_cards += f"""
        <div style='background:white;padding:12px 16px;border-radius:8px;
                    border-left:4px solid {bar_color};margin-bottom:10px'>
          <div style='display:flex;justify-content:space-between;align-items:center'>
            <span style='font-weight:600;color:{COLORS["dark"]};text-transform:capitalize'>
              {dim_name.replace("_", " ")}
            </span>
            <span style='font-weight:700;color:{bar_color}'>
              gap: {gap:.4f}
            </span>
          </div>
          <div style='background:#e2e8f0;border-radius:9999px;height:6px;margin-top:8px;overflow:hidden'>
            <div style='width:{bar_pct:.1f}%;background:{bar_color};height:100%;border-radius:9999px'></div>
          </div>
        </div>"""

    flags_html = ""
    if flags:
        flags_html = "<div style='margin-top:12px'>"
        for f in flags:
            flags_html += f"<div style='padding:6px 12px;background:#fff3cd;border-left:3px solid {COLORS['orange']};margin-bottom:6px;border-radius:4px;font-size:0.85rem'>⚠️ {f}</div>"
        flags_html += "</div>"

    return f"""
    <div style='padding:16px;background:white;border-radius:10px;border:1px solid {COLORS["sage"]}'>
      <div style='display:flex;align-items:center;gap:12px;margin-bottom:16px'>
        <div style='font-size:2rem'>{flag_icon}</div>
        <div>
          <h4 style='margin:0;color:{COLORS["teal"]}'>Bias Audit Report</h4>
          <p style='margin:4px 0 0;color:{flag_color};font-weight:700'>
            Overall Bias Score: {overall:.4f} (lower is better)
          </p>
        </div>
      </div>
      <h5 style='color:{COLORS["teal"]};margin-bottom:8px'>Bias Dimensions</h5>
      {dim_cards}
      {flags_html}
      <p style='margin-top:12px;font-size:0.8rem;color:gray'>
        Threshold: 0.05 — gaps above this threshold are flagged as potential bias.
        Analyzed {len(texts)} samples.
      </p>
    </div>
    """


# ──────────────────────────────────────────────
#  Build Gradio App
# ──────────────────────────────────────────────

_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
)


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Multilingual Indian Sentiment Analysis",
        theme=_theme,
    ) as demo:

        # Header
        gr.HTML(f"""
        <div style='text-align:center;padding:32px 24px 16px;
                    background:linear-gradient(135deg,{COLORS["teal"]},{COLORS["dark"]});
                    border-radius:12px;margin-bottom:24px;color:white'>
          <h1 style='margin:0;font-size:1.8rem;color:white'>
            Multilingual Indian Sentiment Analysis
          </h1>
          <p style='margin:8px 0 0;opacity:0.85;font-size:0.95rem'>
            IndicBERT + LoRA · Hindi · Tamil · Bengali · Marathi · Telugu · English
          </p>
          <div style='display:flex;justify-content:center;gap:16px;margin-top:12px;flex-wrap:wrap'>
            <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:9999px;font-size:0.8rem'>
              F1: 85.1%
            </span>
            <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:9999px;font-size:0.8rem'>
              6 Languages
            </span>
            <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:9999px;font-size:0.8rem'>
              10 Brands
            </span>
            <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:9999px;font-size:0.8rem'>
              ACL 2025 Submission
            </span>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Single Inference ──────────────────────────────────
            with gr.TabItem("🔍 Single Inference"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter Hindi, Tamil, Bengali, Marathi, Telugu, or English text...",
                            lines=4,
                        )
                        with gr.Row():
                            lang_select = gr.Dropdown(
                                choices=list(LANGUAGES.values()),
                                value="Auto-detect",
                                label="Language",
                                scale=2,
                            )
                            sample_select = gr.Dropdown(
                                choices=list(SAMPLE_TEXTS.keys()),
                                label="Load Sample",
                                scale=1,
                            )
                        predict_btn = gr.Button("Analyze Sentiment", variant="primary")

                    with gr.Column(scale=3):
                        sentiment_out    = gr.HTML(label="Sentiment")
                        confidence_out   = gr.HTML(label="Probabilities")
                        stats_out        = gr.HTML(label="Raw Output")

                sample_select.change(fn=load_sample, inputs=sample_select, outputs=text_input)
                predict_btn.click(
                    fn=run_single_inference,
                    inputs=[text_input, lang_select],
                    outputs=[sentiment_out, confidence_out, stats_out],
                )
                text_input.submit(
                    fn=run_single_inference,
                    inputs=[text_input, lang_select],
                    outputs=[sentiment_out, confidence_out, stats_out],
                )

                gr.HTML(f"""
                <div style='margin-top:16px;padding:12px 16px;background:{COLORS["sage"]};
                            border-radius:8px;font-size:0.85rem;color:{COLORS["dark"]}'>
                  <strong>Tip:</strong> The model handles code-mixed text (e.g., Hinglish) automatically.
                  For best results with mixed scripts, leave language on "Auto-detect".
                </div>
                """)

            # ── Tab 2: Batch CSV ─────────────────────────────────────────
            with gr.TabItem("📊 Batch Inference"):
                gr.HTML(f"""
                <p style='color:{COLORS["dark"]}'>
                  Upload a CSV with a <code>text</code> column. Up to 50 rows processed at once.
                </p>
                """)
                with gr.Row():
                    with gr.Column():
                        csv_input  = gr.File(label="Upload CSV", file_types=[".csv"])
                        batch_btn  = gr.Button("Run Batch Inference", variant="primary")
                    with gr.Column(scale=2):
                        batch_out  = gr.HTML(label="Results")
                        dl_file    = gr.File(label="Download Results CSV", visible=True)

                gr.HTML(f"""
                <div style='padding:12px 16px;background:{COLORS["dark"]};border-radius:8px;
                            font-family:monospace;font-size:0.8rem;color:{COLORS["sage"]};margin-top:8px'>
                  # Expected CSV format<br/>
                  text<br/>
                  "जियो का नेटवर्क बहुत तेज़ है"<br/>
                  "Zomato delivery was late again"<br/>
                  "ஃபிளிப்கார்ட் சேவை மிகவும் நன்றாக உள்ளது"
                </div>
                """)

                batch_btn.click(
                    fn=run_batch_inference,
                    inputs=[csv_input],
                    outputs=[batch_out, dl_file],
                )

            # ── Tab 3: Brand Trends ──────────────────────────────────────
            with gr.TabItem("📈 Brand Trends"):
                with gr.Row():
                    brand_select = gr.Dropdown(
                        choices=BRANDS, value="Jio", label="Brand"
                    )
                    days_slider = gr.Slider(
                        minimum=7, maximum=90, value=30, step=1, label="Days"
                    )
                    trends_btn = gr.Button("Load Trends", variant="primary")

                trends_out = gr.HTML()

                trends_btn.click(
                    fn=load_brand_trends,
                    inputs=[brand_select, days_slider],
                    outputs=trends_out,
                )
                brand_select.change(
                    fn=load_brand_trends,
                    inputs=[brand_select, days_slider],
                    outputs=trends_out,
                )

                # Initial load
                demo.load(
                    fn=lambda: load_brand_trends("Jio", 30),
                    outputs=trends_out,
                )

            # ── Tab 4: Bias Audit ────────────────────────────────────────
            with gr.TabItem("⚖️ Bias Audit"):
                gr.HTML(f"""
                <p style='color:{COLORS["dark"]}'>
                  Upload a labeled CSV with columns <code>text</code> and <code>label</code>
                  (0 = positive, 1 = neutral, 2 = negative) to run a full bias audit.
                </p>
                """)
                with gr.Row():
                    with gr.Column():
                        bias_file = gr.File(label="Upload Labeled CSV", file_types=[".csv"])
                        bias_btn  = gr.Button("Run Bias Audit", variant="primary")
                    with gr.Column(scale=2):
                        bias_out = gr.HTML()

                bias_btn.click(
                    fn=run_bias_audit,
                    inputs=[bias_file],
                    outputs=bias_out,
                )

                gr.HTML(f"""
                <div style='margin-top:16px;padding:16px;background:{COLORS["sage"]};
                            border-radius:8px;font-size:0.85rem;color:{COLORS["dark"]}'>
                  <strong style='color:{COLORS["teal"]}'>Bias Dimensions Evaluated:</strong>
                  <ul style='margin:8px 0 0;padding-left:20px'>
                    <li><strong>Gender Bias</strong> — counterfactual tests with gendered pronoun pairs</li>
                    <li><strong>Regional Bias</strong> — metro cities (Mumbai, Delhi) vs tier-2 cities</li>
                    <li><strong>Script Bias</strong> — Devanagari, Latin, Tamil, Bengali, Telugu scripts</li>
                    <li><strong>Brand Bias</strong> — large brands (Jio, Flipkart) vs niche/local brands</li>
                    <li><strong>Class Imbalance</strong> — per-class precision, recall, F1</li>
                  </ul>
                </div>
                """)

        # Footer
        gr.HTML(f"""
        <div style='text-align:center;margin-top:24px;padding:16px;
                    border-top:1px solid {COLORS["sage"]};
                    color:gray;font-size:0.8rem'>
          Multilingual Indian Sentiment Analysis · IndicBERT + LoRA ·
          <a href='https://huggingface.co/ai4bharat/indic-bert' target='_blank'
             style='color:{COLORS["teal"]}'>ai4bharat/indic-bert</a> ·
          ACL 2025 Submission
        </div>
        """)

    return demo


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=20)
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
        theme=_theme,
        css=CUSTOM_CSS,
    )
