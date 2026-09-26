from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COLUMNS = ["title","genre","keywords","description","director","cast"]

class RecommendationEngine:
    def __init__(self, dataframe):
        self.df = dataframe.copy().reset_index(drop=True)
        self._prepare_features()

    @staticmethod
    def _clean(text):
        return re.sub(r"[^a-z0-9\s]", " ", str(text).lower())

    def _prepare_features(self):
        for c in TEXT_COLUMNS:
            self.df[c] = self.df[c].fillna("").astype(str)
        self.df["combined_text"] = self.df[TEXT_COLUMNS].apply(
            lambda row: " ".join(self._clean(x) for x in row), axis=1
        )
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1,2), min_df=1, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(self.df["combined_text"])
        self.similarity_matrix = cosine_similarity(self.matrix)

    def search(self, query, limit=10):
        query = str(query).strip().lower()
        if not query:
            return self.df.head(limit).copy()
        mask = self.df["title"].str.lower().str.contains(query, regex=False, na=False)
        if mask.any():
            return self.df.loc[mask].head(limit).copy()
        qv = self.vectorizer.transform([self._clean(query)])
        scores = cosine_similarity(qv, self.matrix).ravel()
        idx = np.argsort(scores)[::-1][:limit]
        out = self.df.iloc[idx].copy()
        out["search_score"] = scores[idx]
        return out

    def recommend(self, title, n=8, genre_filter=None):
        matches = self.df.index[self.df["title"].str.lower() == str(title).lower()]
        if len(matches) == 0:
            raise ValueError(f"Item not found: {title}")
        idx = int(matches[0])
        scores = self.similarity_matrix[idx].copy()
        scores[idx] = -1
        candidates = self.df.copy()
        candidates["similarity"] = scores
        if genre_filter and genre_filter != "All":
            candidates = candidates[
                candidates["genre"].str.contains(genre_filter, case=False, na=False)
            ]
        candidates = candidates[candidates["similarity"] >= 0]
        if candidates.empty:
            return candidates
        r = candidates["rating"].astype(float)
        p = candidates["popularity"].astype(float)
        rn = (r-r.min())/(r.max()-r.min()+1e-9)
        pn = (p-p.min())/(p.max()-p.min()+1e-9)
        candidates["ranking_score"] = (
            candidates["similarity"]*0.80 + rn*0.12 + pn*0.08
        )
        return candidates.sort_values(
            ["ranking_score","similarity"], ascending=False
        ).head(n).reset_index(drop=True)

    def explain(self, source_title, recommended_title):
        a = self.df[self.df["title"].str.lower() == source_title.lower()]
        b = self.df[self.df["title"].str.lower() == recommended_title.lower()]
        if a.empty or b.empty:
            return "Recommended from overall content similarity."
        ga = set(str(a.iloc[0]["genre"]).lower().split())
        gb = set(str(b.iloc[0]["genre"]).lower().split())
        shared = sorted(ga & gb)
        return ("Shared genre signals: " + ", ".join(shared[:4])) if shared else                "Recommended because its TF-IDF content profile is similar."
