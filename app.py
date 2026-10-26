import streamlit as st
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SmartStats", layout="wide")
st.title("SmartStats – AI Statistical Coach")

@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

df = load_data()

# Load the system prompt
with open("prompts/statistical_coach.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

# Sidebar
st.sidebar.header("Ask a question")
user_question = st.sidebar.text_area(
    "What would you like to understand about this data?",
    placeholder="e.g. How does smoking affect medical charges after controlling for age and BMI?"
)

if st.sidebar.button("Propose Analysis Approach"):
    if user_question.strip():
        st.subheader("AI Statistical Coach")

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",   # or another free Groq model
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.3,
            )
            st.markdown(response.choices[0].message.content)
    else:
        st.warning("Please type a question first.")

# Main content
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Basic Info")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.write(df.describe())