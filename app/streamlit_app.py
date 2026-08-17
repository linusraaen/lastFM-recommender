"""Streamlit demo: type any Last.fm username, get live artist recommendations.

Talks to the FastAPI service (src/serve/api.py) over HTTP so it exercises the
same serving path a real client would -- set API_URL if it's not local.

    API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Last.fm Two-Tower Recommender", page_icon="🎧")
st.title("🎧 Two-Tower Music Recommender")
st.caption("Trained on a self-collected Last.fm dataset. Enter any Last.fm username.")

with st.form("recommend_form"):
    username = st.text_input("Last.fm username", value="")
    k = st.slider("Number of recommendations", min_value=5, max_value=50, value=20)
    submitted = st.form_submit_button("Recommend")

if submitted:
    if not username.strip():
        st.warning("Enter a username first.")
    else:
        with st.spinner(f"Fetching {username}'s listening history and ranking artists..."):
            try:
                resp = requests.get(f"{API_URL}/recommend", params={"user": username, "k": k}, timeout=30)
            except requests.RequestException as e:
                st.error(f"Couldn't reach the recommender API at {API_URL}: {e}")
                resp = None

        if resp is not None:
            if resp.status_code == 404:
                st.error(f"No public listening history found for '{username}'.")
            elif resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                if data["cold_start"]:
                    st.info("No overlap with the training catalogue -- showing popular-artist fallback.")
                st.caption(f"Served in {data['latency_ms']:.0f} ms")

                for rec in data["recommendations"]:
                    tags = ", ".join(rec["tags"]) if rec["tags"] else "—"
                    st.markdown(f"**{rec['artist']}**  \n:musical_note: {tags}")
                    st.divider()

st.sidebar.header("About")
st.sidebar.markdown(
    "Two-tower retrieval model: artist embeddings (genre tags + bio text + "
    "popularity) precomputed and indexed in FAISS; the user tower runs per "
    "request over your live top-artists to build a query embedding, then an "
    "ANN lookup returns ranked artists."
)
