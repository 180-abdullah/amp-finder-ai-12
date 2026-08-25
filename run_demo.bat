@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -e .
python scripts\create_demo_assets.py
streamlit run app.py
endlocal
