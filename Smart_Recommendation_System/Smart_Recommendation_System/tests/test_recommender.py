import pandas as pd
from src.recommender import RecommendationEngine

def data():
    return pd.DataFrame([
        {"id":1,"title":"Space Mission","genre":"Sci-Fi Adventure","keywords":"space astronaut planet","description":"An astronaut explores a distant planet.","director":"A","cast":"X","rating":8.5,"popularity":90},
        {"id":2,"title":"Galaxy Journey","genre":"Sci-Fi Adventure","keywords":"space astronaut galaxy","description":"A crew travels across a galaxy.","director":"B","cast":"Y","rating":8.2,"popularity":85},
        {"id":3,"title":"Cooking Home","genre":"Drama","keywords":"family food kitchen","description":"A family rebuilds life through cooking.","director":"C","cast":"Z","rating":7.2,"popularity":70},
    ])

def test_count():
    r = RecommendationEngine(data()).recommend("Space Mission", 2)
    assert len(r) == 2
    assert "similarity" in r.columns

def test_excludes_source():
    r = RecommendationEngine(data()).recommend("Space Mission", 5)
    assert "Space Mission" not in r["title"].tolist()

def test_search():
    r = RecommendationEngine(data()).search("Galaxy")
    assert r.iloc[0]["title"] == "Galaxy Journey"
