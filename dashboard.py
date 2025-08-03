### dashboard.py
import streamlit as st
import pandas as pd
from PIL import Image

df = pd.read_csv("qa_report.csv")

st.title("Diffusion Model QA Dashboard")

for _, row in df.iterrows():
    st.subheader(row["Prompt"])
    st.image(row["Image Path"], width=256)
    st.write(f"CLIP Similarity: {row['CLIP Similarity']:.2f}")
