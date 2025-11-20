"""
MLA3 - Michigan Lottery Analyzer v3.0

Enhanced prediction and analysis powered by an ensemble of statistical
signals over historical CSV data. Fast, explainable, and production-friendly.
"""

import os
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

from fixed_scraper import ProductionMichiganScraper
from src.analysis.mla3_predictor import predict, DEFAULT_SPECS, df_from_history


st.set_page_config(page_title="MLA3 - Michigan Lottery Analyzer v3", layout="wide")


@st.cache_data(show_spinner=False)
def load_history(game: str, draws: int = 365) -> List[Dict[str, Any]]:
    scraper = ProductionMichiganScraper()
    hist = scraper.get_recent_history(game, count=draws)
    return hist


def render_header():
    st.title("MLA3 - Michigan Lottery Analyzer v3.0")
    st.caption("Ensemble predictions using frequency, recency, due-gap, and pairwise association with sum-band constraints")


def render_sidebar():
    st.sidebar.header("Configuration")
    games = list(DEFAULT_SPECS.keys())
    game = st.sidebar.selectbox("Game", games, index=games.index("Powerball") if "Powerball" in games else 0)
    draws = st.sidebar.slider("History size (draws)", 50, 1000, 365, step=25)
    n_sets = st.sidebar.slider("Prediction sets", 1, 10, 5)
    return game, draws, n_sets


def render_predictions(pred: Dict[str, Any]):
    st.subheader("Predicted Number Sets")
    cols = st.columns(min(5, max(1, len(pred.get("sets", [])))))
    for i, s in enumerate(pred.get("sets", [])):
        with cols[i % len(cols)]:
            st.metric(f"Set {i+1}", ", ".join(map(str, s["numbers"])), f"Conf: {int(100*s['confidence'])}%")
            if s.get("bonus") is not None:
                st.caption(f"Bonus: {s['bonus']}")
    if pred.get("explanations"):
        with st.expander("How these were generated"):
            for line in pred["explanations"]:
                st.write("- ", line)


def render_analysis(df: pd.DataFrame, game: str):
    st.subheader("Analysis")
    if df.empty:
        st.info("No history available.")
        return
    # flatten numbers
    all_nums: List[int] = []
    for _, row in df.iterrows():
        all_nums += row["numbers"]
    if not all_nums:
        st.info("No numbers in history.")
        return

    freq = pd.Series(all_nums).value_counts().sort_index()
    c1, c2 = st.columns(2)
    with c1:
        st.write("Top frequencies")
        st.bar_chart(freq)
    with c2:
        st.write("Recent sums")
        sums = [sum(x) for x in df["numbers"] if x]
        st.line_chart(pd.Series(sums))


def main():
    render_header()
    game, draws, n_sets = render_sidebar()

    with st.spinner("Loading history..."):
        history = load_history(game, draws)
        df = df_from_history(history)

    if not history:
        st.error("No historical data available for this game.")
        return

    with st.spinner("Generating predictions..."):
        pred = predict(history, game, n_sets=n_sets)

    render_predictions(pred)
    st.divider()
    render_analysis(df, game)


if __name__ == "__main__":
    main()
