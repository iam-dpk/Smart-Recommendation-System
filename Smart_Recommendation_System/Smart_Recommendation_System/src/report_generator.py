from pathlib import Path
from datetime import datetime
from .data_loader import load_dataset
from .recommender import RecommendationEngine
from .evaluation import demo_evaluation

def generate_report(output=None):
    df = load_dataset()
    engine = RecommendationEngine(df)
    m = demo_evaluation(engine)
    if output is None:
        output = Path(__file__).resolve().parent.parent/"reports"/"recommendation_report.md"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = f'''# Smart Recommendation System Report

Generated: {datetime.now():%Y-%m-%d %H:%M}

## Dataset
- Items: {len(df)}
- Average rating: {df["rating"].mean():.2f}
- Average popularity: {df["popularity"].mean():.2f}
- Unique genres: {df["genre"].nunique()}

## Model
- TF-IDF vectorization
- Unigrams and bigrams
- Cosine similarity
- Rating/popularity ranking adjustment

## Offline Proxy Evaluation
- Samples: {m["samples"]}
- K: {m["k"]}
- Genre alignment@K: {m["genre_alignment_at_k"]:.2%}

> This is a lightweight offline proxy because the demo dataset has no real user interaction history.

## Future Work
Collaborative filtering, hybrid ranking, user profiles, implicit feedback, neural ranking, and online evaluation.
'''
    output.write_text(report, encoding="utf-8")
    return output

if __name__ == "__main__":
    print(f"Report created: {generate_report()}")
