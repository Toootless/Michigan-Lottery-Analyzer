import os
import time
from typing import List

import streamlit as st
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
    _OPENAI_IMPORTED = True
except Exception:
    _OPENAI_IMPORTED = False

from src.utils.helpers import get_project_version

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Lottery AI Workbench", page_icon="🎰", layout="wide")

st.title("🎰 Lottery AI Workbench")
st.caption(f"Version {get_project_version()} • Python {os.sys.version.split()[0]}")

# Sidebar controls
game = st.sidebar.selectbox(
    "Select Game", ["Powerball", "Mega Millions", "Fantasy 5", "Daily 3", "Daily 4"], index=0
)
window = st.sidebar.slider("Days of History (synthetic)", min_value=30, max_value=365, value=90, step=15)
st.sidebar.write("Data source: synthetic placeholder")

# Synthetic frequency data generator
def generate_dummy_frequency(game_name: str, days: int) -> pd.DataFrame:
    import random
    random.seed(hash(game_name) % 10_000 + days)
    # vary range by game (simplified)
    if game_name in ("Powerball", "Mega Millions"):
        size = 69
    elif game_name == "Fantasy 5":
        size = 39
    elif game_name == "Daily 3":
        size = 10
    else:
        size = 10
    numbers = list(range(1, size + 1))
    freq = [random.randint(0, 20) for _ in numbers]
    return pd.DataFrame({"number": numbers, "frequency": freq}).sort_values("frequency", ascending=False).head(25)

freq_df = generate_dummy_frequency(game, window)
fig = px.bar(freq_df, x="number", y="frequency", title=f"Top {len(freq_df)} Frequencies ({game})")
fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

# Placeholder ML prediction function
def predict_numbers(game_name: str, top_df: pd.DataFrame, count: int = 5) -> List[int]:
    # naive selection: take highest frequency then shuffle slighty
    base = top_df.sort_values("frequency", ascending=False)["number"].tolist()
    return base[:count]

with st.expander("🎯 Generate Sample Prediction", expanded=True):
    num_count = 5 if game not in ("Daily 3", "Daily 4") else (3 if game == "Daily 3" else 4)
    predicted = predict_numbers(game, freq_df, num_count)
    st.write(f"Suggested Numbers: {predicted}")
    st.caption("This is a placeholder algorithm. Replace with real model logic.")

# Optional OpenAI Chat panel (only when key & library available)
if OPENAI_KEY and _OPENAI_IMPORTED:
    st.subheader("🤖 OpenAI Chat")
    client = OpenAI(api_key=OPENAI_KEY)
    user_q = st.text_input("Ask about the synthetic frequencies:")
    if user_q:
        with st.spinner("Querying OpenAI..."):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an assistant summarizing dummy lottery frequency data."},
                        {"role": "user", "content": user_q},
                    ],
                    max_tokens=200,
                )
                st.write(completion.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {e}")
else:
    st.info("Set OPENAI_API_KEY in .env to enable chat panel.")

st.markdown("---")
st.caption("LotteryAIWorkbench • Replace placeholders with real data ingestion & modeling.")
