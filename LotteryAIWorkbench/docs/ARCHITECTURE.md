# LotteryAIWorkbench Architecture

## Overview
LotteryAIWorkbench is a modular scaffold for building lottery analytics features with incremental machine learning enhancements. It intentionally starts simple: synthetic data generation flows into the UI, then is replaced by real ingestion, feature engineering, and prediction pipelines.

## Layers
1. Data Layer (`src/data/`): Future modules for fetching & normalizing draw history from external APIs or CSV archives.
2. ML Layer (`src/ml/`): Model training, feature engineering, evaluation reports. Initially empty; add notebooks or Python modules here.
3. Utils Layer (`src/utils/`): Small cross-cutting helpers (version access, caching, validation).
4. App Layer (`app/`): Streamlit front-end; orchestrates frequency summarization, model inference, optional LLM assistance.
5. Scripts Layer (`scripts/`): Operational helpers (development runner, data seeding, batch jobs).
6. Tests (`tests/`): Pytest-based validation (smoke tests now, deeper scenario tests later).

## Data Flow (Initial Placeholder)
Synthetic frequency -> bar chart -> naive top-N prediction -> (optional) explanation via OpenAI.

## Planned Evolution
- Replace synthetic generator with a `DataIngestor` class pulling historical draws and computing frequency & recency features.
- Add `FeatureBuilder` for statistical and engineered features (hot/cold streaks, delta spacing, positional entropy).
- Introduce `ModelRegistry` to manage candidate ML models (e.g., probabilistic number ranking) with versioned artifacts.
- Add evaluation dashboard: lift charts, calibration plots, simulation backtesting results.

## Extensibility Guidelines
- Favor pure functions for transformations; isolate I/O.
- Keep UI logic thin—delegate computation to `src/` modules.
- Use environment variables for secrets (API keys) through `python-dotenv`.
- Gate optional dependencies (OpenAI) gracefully when not installed or configured.

## Versioning
`get_project_version()` provides a unified version string; update `_FALLBACK_VERSION` when incrementing scaffold version until packaging.

## Future Security & Compliance
- Input validation for external data sources.
- Rate limiting for API calls.
- Transparent provenance logging for generated predictions.

## Testing Strategy
- Phase 1: Smoke tests (imports & version).
- Phase 2: Unit tests for frequency calculations & feature engineering.
- Phase 3: Model evaluation tests ensuring performance thresholds.
- Phase 4: UI interaction tests (Streamlit components via playwright or snapshot strategies).
