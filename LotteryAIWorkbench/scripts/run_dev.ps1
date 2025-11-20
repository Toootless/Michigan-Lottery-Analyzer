Param(
    [int]$Port = 8610
)

$env:PYTHONUNBUFFERED = 1
Write-Host "Launching Streamlit app on port $Port" -ForegroundColor Cyan
streamlit run "$PSScriptRoot/../app/streamlit_app.py" --server.port $Port --server.headless true