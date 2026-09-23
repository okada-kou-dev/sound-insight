$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8506
if ($LASTEXITCODE -ne 0) { throw "App failed: $LASTEXITCODE" }
