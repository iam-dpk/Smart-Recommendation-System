import numpy as np

def precision_at_k(recommended_ids, relevant_ids, k=5):
    rec = list(recommended_ids)[:k]
    rel = set(relevant_ids)
    return sum(x in rel for x in rec)/len(rec) if rec else 0.0

def recall_at_k(recommended_ids, relevant_ids, k=5):
    rel = set(relevant_ids)
    if not rel: return 0.0
    return len(set(list(recommended_ids)[:k]) & rel)/len(rel)

def diversity(recommendations, genre_column="genre"):
    if recommendations.empty or genre_column not in recommendations:
        return 0.0
    return recommendations[genre_column].nunique()/len(recommendations)

def demo_evaluation(engine, sample_size=10, k=5):
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(engine.df), size=min(sample_size,len(engine.df)), replace=False)
    values = []
    for idx in idxs:
        source = engine.df.iloc[idx]
        recs = engine.recommend(source["title"], n=k)
        sg = set(str(source["genre"]).lower().split())
        hits = sum(bool(sg & set(str(g).lower().split())) for g in recs["genre"])
        values.append(hits/max(len(recs),1))
    return {
        "samples": len(values),
        "genre_alignment_at_k": float(np.mean(values)) if values else 0.0,
        "k": k,
        "note": "Offline proxy metric based on shared genre tokens."
    }
