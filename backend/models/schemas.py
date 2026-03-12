"""Pydantic request/response schemas for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single-sample sentiment prediction request.

    Attributes:
        text: Raw input text (1–512 characters).
        language: BCP-47 language hint (optional; auto-detected if omitted).
    """

    text: str = Field(..., min_length=1, max_length=512)
    language: str | None = Field(
        default=None,
        pattern=r"^(hi|ta|bn|te|mr|code_mix)$",
        description="ISO language code or 'code_mix'",
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class PredictBatchRequest(BaseModel):
    """Batch sentiment prediction request.

    Attributes:
        texts: List of 1–50 input texts.
        language: Optional language override applied to all samples.
    """

    texts: list[str] = Field(..., min_length=1, max_length=50)
    language: str | None = None

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        stripped = [t.strip() for t in v]
        if any(len(t) == 0 for t in stripped):
            raise ValueError("Empty strings are not allowed in batch.")
        if any(len(t) > 512 for t in stripped):
            raise ValueError("All texts must be ≤ 512 characters.")
        return stripped


class SentimentLabel(BaseModel):
    """A single predicted sentiment with confidence scores.

    Attributes:
        label: Human-readable sentiment label.
        label_id: Integer label index (0=positive, 1=neutral, 2=negative).
        confidence: Probability of the predicted class.
        probabilities: Per-class probability distribution.
        language_detected: Detected language code.
        inference_ms: Server-side inference time in milliseconds.
    """

    label: str
    label_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    language_detected: str | None = None
    inference_ms: float = 0.0


class PredictResponse(BaseModel):
    """Response for a single prediction."""

    text: str
    prediction: SentimentLabel
    model_version: str = "1.0.0"


class BatchPredictResponse(BaseModel):
    """Response for batch prediction."""

    results: list[PredictResponse]
    total: int
    batch_inference_ms: float = 0.0


class TrendPoint(BaseModel):
    """A single time-series data point.

    Attributes:
        date: ISO date string (YYYY-MM-DD).
        brand: Brand name.
        positive_pct: % positive samples.
        neutral_pct: % neutral samples.
        negative_pct: % negative samples.
        volume: Total sample count for the interval.
    """

    date: str
    brand: str
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    volume: int


class TrendsResponse(BaseModel):
    """Response for GET /trends."""

    brand: str
    days: int
    data: list[TrendPoint]
    total_samples: int


class BiasCheckRequest(BaseModel):
    """Request to run bias audit on a batch of samples.

    Attributes:
        texts: Sample texts.
        labels: Ground-truth integer labels.
        brands: Per-sample brand strings.
        languages: Per-sample language codes.
    """

    texts: list[str] = Field(..., min_length=5)
    labels: list[int]
    brands: list[str] | None = None
    languages: list[str] | None = None


class BiasCheckResponse(BaseModel):
    """Response from the bias audit endpoint."""

    overall_bias_score: float
    gender_bias_score: float
    regional_gap: float
    script_gap: float
    brand_gap: float
    class_precision: dict[str, float]
    class_recall: dict[str, float]
    bias_flags: list[str]
    sample_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    cache_alive: bool
    gpu_available: bool
    version: str = "1.0.0"
