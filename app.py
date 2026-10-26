import streamlit as st
import pandas as pd

st.set_page_config(page_title="QuantPilot", layout="wide")
st.title("QuantPilot – AI Statistical Coach")

@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Basic Info")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.write(df.describe())
