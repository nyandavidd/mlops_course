@echo off
echo Starting Python HTTP server...
echo Access heatmap at: http://localhost:8000/heatmap.html
echo Press Ctrl+C to stop the server
cd %~dp0
python -m http.server 8000
pause 