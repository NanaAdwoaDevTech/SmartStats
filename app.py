import streamlit as st
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
import statsmodels.formula.api as smf
import plotly.express as px

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
                model="qwen/qwen3.8-27b",
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

def run_smoking_analysis(df):
    """Simple, transparent analysis for the smoking question"""
    
    # Prepare data
    df = df.copy()
    df["smoker_bin"] = (df["smoker"] == "yes").astype(int)
    
    # Model: charges ~ smoker + age + bmi
    model = smf.ols("charges ~ smoker_bin + age + bmi", data=df).fit()
    
    # Key results
    smoker_effect = model.params["smoker_bin"]
    p_value = model.pvalues["smoker_bin"]
    
    # Chart 1: Average charges by smoker status
    fig1 = px.box(df, x="smoker", y="charges", 
                  title="Medical Charges by Smoking Status",
                  labels={"smoker": "Smoker", "charges": "Charges ($)"})
    
    # Chart 2: Charges vs Age, colored by smoker
    fig2 = px.scatter(df, x="age", y="charges", color="smoker",
                      title="Charges vs Age by Smoking Status",
                      labels={"age": "Age", "charges": "Charges ($)"},
                      opacity=0.6)
    
    return {
        "smoker_effect": smoker_effect,
        "p_value": p_value,
        "r_squared": model.rsquared,
        "summary": model.summary().as_text(),
        "fig1": fig1,
        "fig2": fig2
    }

if st.sidebar.button("Run Analysis & Show Charts"):
    st.subheader("Analysis Results")
    
    results = run_smoking_analysis(df)
    
    st.metric("Estimated extra charge for smokers (controlling for age & BMI)", 
              f"${results['smoker_effect']:,.0f}")
    st.write(f"P-value: {results['p_value']:.4f}")
    st.write(f"R-squared: {results['r_squared']:.3f}")
    
    st.plotly_chart(results["fig1"], use_container_width=True)
    st.plotly_chart(results["fig2"], use_container_width=True)
    
    with st.expander("Full model summary"):
        st.text(results["summary"])