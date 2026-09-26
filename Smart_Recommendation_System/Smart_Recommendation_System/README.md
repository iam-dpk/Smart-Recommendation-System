# Smart Recommendation System

A portfolio-ready AI recommendation system built with Python, scikit-learn, and Streamlit.

## Features
- Content-based recommendations using TF-IDF + cosine similarity
- Ranking adjustment using rating and popularity
- Searchable Streamlit dashboard
- Recommendation explanations and similarity scores
- CSV export
- Offline evaluation utility
- Synthetic dataset included for immediate offline use
- Unit tests and Windows launchers
- Report generator

## Run on Windows
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Or double-click `run_windows.bat`.

PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_windows.ps1
```

## Tests
```bash
python -m pytest -q
```

## Generate report
```bash
python -m src.report_generator
```

## How it works
1. Text fields are cleaned and combined.
2. TF-IDF converts each item into a feature vector.
3. Cosine similarity finds related items.
4. Content similarity is the dominant ranking signal, with small rating/popularity adjustments.
5. Streamlit presents the recommendations.

## Dataset
The included 100-item catalog is synthetic demonstration data created specifically for this project, so it runs without an external API.

## Future upgrades
Collaborative filtering, hybrid recommendations, user profiles, feedback loops, neural ranking, and deployment.

## Author
Deepak Kumar Shukla

Portfolio: https://deepakshukla.online
GitHub: https://github.com/iam-dpk
LinkedIn: https://www.linkedin.com/in/dk-50s
