# LotteryAIWorkbench

A minimal Python 3.11 + Streamlit AI analytics workbench for exploring lottery number patterns.

## Features
- Streamlit UI with sidebar game selector
- Dummy frequency visualization (Plotly bar chart)
- Placeholder ML prediction function (scikit-learn ready)
- Optional OpenAI chat section (enabled when `OPENAI_API_KEY` is set)
- Simple utilities module with version helper
- Ready for expansion (data ingestion, model training, evaluation)

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py --server.port 8610
```

## Environment
Create a `.env` file based on `.env.example`:
```
OPENAI_API_KEY=sk-... # optional
```
If the key is missing, the chat panel is hidden.

## Project Structure
```
LotteryAIWorkbench/
  app/streamlit_app.py
  src/
    data/
    ml/
    utils/helpers.py
  tests/test_smoke.py
  scripts/run_dev.ps1
  scripts/seed_data.py
  docs/ARCHITECTURE.md
  docs/API.md
  requirements.txt
  pyproject.toml
  .env.example
  LICENSE
```

## Development Scripts
PowerShell dev script:
```powershell
./scripts/run_dev.ps1
```
Launches Streamlit on port 8610.

## Placeholder Notes
- `seed_data.py` contains stub logic; replace with real ingestion.
- ML section uses a dummy predictor; extend with actual training pipelines.

## License
MIT
