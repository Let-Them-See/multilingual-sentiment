import { API_V1 } from "./constants";
import type {
  BatchPredictResponse,
  BiasCheckResponse,
  HealthResponse,
  PredictResponse,
  TrendsResponse,
} from "./types";

// ── Shared fetch wrapper ───────────────────────────────────────────────────
async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_V1}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ── Predict ────────────────────────────────────────────────────────────────
export async function predictSingle(
  text: string,
  language?: string,
): Promise<PredictResponse> {
  return apiFetch<PredictResponse>("/predict/", {
    method: "POST",
    body: JSON.stringify({ text, language: language ?? null }),
  });
}

export async function predictBatch(
  texts: string[],
  language?: string,
): Promise<BatchPredictResponse> {
  return apiFetch<BatchPredictResponse>("/predict/batch", {
    method: "POST",
    body: JSON.stringify({ texts, language: language ?? null }),
  });
}

// ── Trends ─────────────────────────────────────────────────────────────────
export async function getTrends(
  brand: string,
  days = 30,
): Promise<TrendsResponse> {
  return apiFetch<TrendsResponse>(
    `/trends/?brand=${encodeURIComponent(brand)}&days=${days}`,
  );
}

export async function getBrands(): Promise<string[]> {
  return apiFetch<string[]>("/trends/brands");
}

// ── Bias ───────────────────────────────────────────────────────────────────
export async function checkBias(
  texts: string[],
  labels: number[],
  brands?: string[],
  languages?: string[],
): Promise<BiasCheckResponse> {
  return apiFetch<BiasCheckResponse>("/bias/check", {
    method: "POST",
    body: JSON.stringify({ texts, labels, brands, languages }),
  });
}

// ── Health ─────────────────────────────────────────────────────────────────
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_V1.replace("/api/v1", "")}/health`);
  return res.json() as Promise<HealthResponse>;
}
