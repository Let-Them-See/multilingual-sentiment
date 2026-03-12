// ── API Response Types ─────────────────────────────────────────────────────

export interface SentimentLabel {
  label: "positive" | "neutral" | "negative";
  label_id: 0 | 1 | 2;
  confidence: number;
  probabilities: {
    positive: number;
    neutral: number;
    negative: number;
  };
  language_detected: string | null;
  inference_ms: number;
}

export interface PredictResponse {
  text: string;
  prediction: SentimentLabel;
  model_version: string;
}

export interface BatchPredictResponse {
  results: PredictResponse[];
  total: number;
  batch_inference_ms: number;
}

export interface TrendPoint {
  date: string;
  brand: string;
  positive_pct: number;
  neutral_pct: number;
  negative_pct: number;
  volume: number;
}

export interface TrendsResponse {
  brand: string;
  days: number;
  data: TrendPoint[];
  total_samples: number;
}

export interface BiasCheckResponse {
  overall_bias_score: number;
  gender_bias_score: number;
  regional_gap: number;
  script_gap: number;
  brand_gap: number;
  class_precision: Record<string, number>;
  class_recall: Record<string, number>;
  bias_flags: string[];
  sample_count: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  cache_alive: boolean;
  gpu_available: boolean;
  version: string;
}

// ── UI Component Types ─────────────────────────────────────────────────────

export type SentimentType = "positive" | "neutral" | "negative";

export interface LanguageOption {
  value: string;
  label: string;
}

export interface AblationRow {
  config: string;
  f1_macro: number;
  f1_hindi: number;
  f1_tamil: number;
  f1_bengali: number;
  f1_code_mix: number;
  params_m: number;
  is_best: boolean;
}
