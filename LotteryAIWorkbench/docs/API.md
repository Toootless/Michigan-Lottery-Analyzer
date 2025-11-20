# LotteryAIWorkbench Planned API Surface

This document outlines the anticipated Python and potential HTTP interfaces as the project matures. Current implementation is limited to internal helpers and the Streamlit UI.

## Python Modules (Planned)

### src/data/ingestor.py
- `class DataIngestor`: Fetches raw draws from sources (lottery API, CSV, PDF OCR).
  - `fetch(date_from, date_to) -> DataFrame`
  - `normalize(df) -> DataFrame`
  - `compute_frequency(df) -> DataFrame`

### src/ml/features.py
- `build_features(draw_df) -> DataFrame`: Adds engineered columns (recency, gaps, streaks).

### src/ml/model.py
- `train(feature_df, config) -> ModelArtifact`
- `predict(model, latest_features, top_n=5) -> List[int]`
- `evaluate(model, test_features) -> Dict[str, float]`

### src/ml/registry.py
- `register(artifact) -> str`: Persist model with metadata.
- `load(model_id) -> ModelArtifact`

### src/utils/cache.py (optional)
- Simple on-disk caching using `pickle` or `joblib`.

## Streamlit App Functions (Current)
- `predict_numbers(game_name: str, top_df: DataFrame, count: int) -> List[int]`: Naive selection helper.

## Potential REST / GraphQL (Future)
If an API layer is desired (FastAPI or similar):

### Endpoints (FastAPI example)
- `GET /health`: Basic liveness probe.
- `GET /games`: List supported games.
- `GET /frequency/{game}`: Return latest frequency summary JSON.
- `POST /predict`: Accept game + optional strategy; return predicted numbers.
- `GET /model/{id}/metrics`: Retrieve evaluation metadata.

### Authentication (Future)
- API key header (`X-API-Key`) or OAuth2 for secured endpoints.

## Response Schema Examples (Future)
`/frequency/{game}`:
```json
{
  "game": "Powerball",
  "generated_at": "2025-11-13T10:15:00Z",
  "top": [
    {"number": 23, "frequency": 14},
    {"number": 7, "frequency": 13}
  ]
}
```

`/predict`:
```json
{
  "game": "Powerball",
  "strategy": "naive-top",
  "numbers": [23, 7, 18, 42, 11],
  "version": "0.1.0"
}
```

## Versioning & Compatibility
- Semantic versioning for Python package.
- Backwards-compatible additions favored; breaking changes batched into minor/major increments.

## Error Handling (Planned)
- Structured error objects: `{ "error": { "type": "ValidationError", "message": "..." } }`.
- Clear distinction between client (4xx) and server (5xx) failures.

## Observability (Planned)
- Logging: `structlog` or standard library with JSON formatting.
- Metrics: Integration with Prometheus or OpenTelemetry for API latency, model inference counts.

## Rate Limiting (Future)
- Token bucket or external gateway (e.g., API Management) for public endpoints.

## Extensibility Notes
- Keep business logic in pure functions to ease future wrapping as HTTP endpoints.
- Provide clear data contracts (pydantic models) once stability improves.
