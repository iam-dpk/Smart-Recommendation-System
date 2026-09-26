import pandas as pd
import streamlit as st
from src.data_loader import load_dataset
from src.recommender import RecommendationEngine
from src.evaluation import demo_evaluation

st.set_page_config(page_title="Smart Recommendation System", page_icon="🎬", layout="wide")

@st.cache_data
def get_data():
    return load_dataset()

@st.cache_resource
def get_engine():
    return RecommendationEngine(get_data())

df = get_data()
engine = get_engine()

st.markdown("""
<style>
.hero {padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(135deg,#171c2b,#202746);border:1px solid #303957;margin-bottom:1rem}
.card {padding:1rem;border-radius:15px;background:#151a24;border:1px solid #2b3244;min-height:170px}
.muted {color:#9aa4b2}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🎬 Smart Recommendation System</h1>
<p class="muted">AI-powered content recommendations using TF-IDF, cosine similarity, and lightweight ranking signals.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Recommendation Controls")
    title = st.selectbox("Choose an item", df["title"].tolist())
    n = st.slider("Number of recommendations", 3, 15, 8)
    genres = ["All"] + sorted({x.strip() for value in df["genre"] for x in str(value).split() if x.strip()})
    genre_filter = st.selectbox("Optional genre filter", genres)
    st.divider()
    st.subheader("Dataset")
    st.metric("Items", len(df))
    st.metric("Average rating", f'{df["rating"].mean():.2f}')

tab1, tab2, tab3 = st.tabs(["✨ Recommendations", "🔎 Search", "📊 Analytics"])

with tab1:
    selected = df[df["title"] == title].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Selected", selected["title"])
    c2.metric("Rating", f'{selected["rating"]:.1f}/10')
    c3.metric("Popularity", f'{selected["popularity"]:.0f}/100')
    c4.metric("Genre", selected["genre"])
    st.subheader("Recommended for you")
    results = engine.recommend(title, n=n, genre_filter=genre_filter)
    if results.empty:
        st.warning("No items match that filter.")
    else:
        cols = st.columns(2)
        for i,(_,row) in enumerate(results.iterrows()):
            with cols[i%2]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### {i+1}. {row['title']}")
                st.write(f"**Genre:** {row['genre']}")
                st.write(f"**Rating:** {row['rating']:.1f}/10  •  **Popularity:** {row['popularity']:.0f}/100")
                st.progress(min(max(float(row["similarity"]),0),1))
                st.caption(f"Content similarity: {row['similarity']:.1%}")
                st.caption(engine.explain(title,row["title"]))
                st.markdown("</div>", unsafe_allow_html=True)
        cols = ["id","title","genre","rating","popularity","similarity","ranking_score"]
        st.download_button("⬇️ Download recommendations as CSV",
            data=results[cols].to_csv(index=False).encode("utf-8"),
            file_name="recommendations.csv", mime="text/csv")

with tab2:
    query = st.text_input("Search the catalog", placeholder="Try: space, detective, family...")
    if query:
        sr = engine.search(query, limit=12)
        if sr.empty: st.info("No matching items found.")
        else: st.dataframe(sr[["title","genre","rating","popularity"]], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Project Analytics")
    a,b,c = st.columns(3)
    a.metric("Catalog size", len(df))
    b.metric("Unique genres", df["genre"].nunique())
    c.metric("Average rating", f'{df["rating"].mean():.2f}')
    st.bar_chart(df.groupby("genre")["rating"].mean().sort_values(ascending=False))
    m = demo_evaluation(engine)
    st.metric("Genre alignment@K", f'{m["genre_alignment_at_k"]:.1%}')
    st.caption(m["note"])

st.divider()
st.caption("Built as a portfolio-ready demonstration project • Deepak Kumar Shukla")
