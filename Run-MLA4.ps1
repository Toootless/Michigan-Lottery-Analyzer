param(
  [int]$Port = 8505
)

Write-Host "Starting MLA4 on port $Port..." -ForegroundColor Cyan

# Optional: open browser yourself after a short delay (Streamlit also tries automatically when not headless)
Start-Job -ScriptBlock {
  param($p)
  Start-Sleep -Seconds 3
  Start-Process "http://localhost:$p"
} -ArgumentList $Port | Out-Null

# Run Streamlit with non-headless mode so it auto-opens
streamlit run MLA4.py --server.headless false --server.port $Port
