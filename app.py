# ============================================================
# 1. Imports
# ============================================================
import streamlit as st
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
import statsmodels.formula.api as smf
import plotly.express as px

# ============================================================
# 2. load_dotenv() + config
# ============================================================
load_dotenv()  # Loads secrets from .env (never hard-code keys)

st.set_page_config(page_title="SmartStats", layout="wide")
st.title("SmartStats – AI Statistical Coach")

# ============================================================
# 3. Cached data loading
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

df = load_data()

# Load the system prompt (keeps the main file clean)
with open("prompts/statistical_coach.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

# ============================================================
# 4. Helper functions
# ============================================================
def run_analysis(df, question: str):
    """
    Run a suitable analysis based on keywords in the question.
    Returns a dictionary with results + charts, or None if unsupported.
    """
    df = df.copy()  # Never modify the original dataframe
    df["smoker_bin"] = (df["smoker"] == "yes").astype(int)
    
    q = question.lower()
    
    # ----- Case 1: Smoking + controls -----
    if "smok" in q and ("age" in q or "bmi" in q):
        model = smf.ols("charges ~ smoker_bin + age + bmi", data=df).fit()
        effect = model.params["smoker_bin"]
        title = "Estimated extra charge for smokers (controlling for age & BMI)"
        fig1 = px.box(df, x="smoker", y="charges", title="Charges by Smoking Status")
        fig2 = px.scatter(df, x="age", y="charges", color="smoker",
                          title="Charges vs Age by Smoking Status", opacity=0.6)

    # ----- Case 2: Simple smoking difference -----
    elif "smok" in q:
        model = smf.ols("charges ~ smoker_bin", data=df).fit()
        effect = model.params["smoker_bin"]
        title = "Average extra charge for smokers"
        fig1 = px.box(df, x="smoker", y="charges", title="Charges by Smoking Status")
        fig2 = None

    # ----- Case 3: BMI effect -----
    elif "bmi" in q:
        model = smf.ols("charges ~ bmi + age + smoker_bin", data=df).fit()
        effect = model.params["bmi"]
        title = "Estimated change in charges per 1 unit of BMI"
        fig1 = px.scatter(df, x="bmi", y="charges", color="smoker",
                          title="Charges vs BMI", opacity=0.6, trendline="ols")
        fig2 = None

    # ----- Case 4: Region (real averages) -----
    elif "region" in q:
        region_means = df.groupby("region")["charges"].mean().sort_values(ascending=False)
        highest = region_means.index[0]
        lowest = region_means.index[-1]
        effect = region_means.iloc[0] - region_means.iloc[-1]
        
        title = f"Highest region ({highest}): ${region_means.iloc[0]:,.0f}  |  Lowest ({lowest}): ${region_means.iloc[-1]:,.0f}"
        
        fig1 = px.box(df, x="region", y="charges", 
                      title="Medical Charges by Region",
                      category_orders={"region": region_means.index.tolist()})
        fig2 = None
        model = smf.ols("charges ~ C(region)", data=df).fit()

    # ----- Case 5: Children (real averages) -----
    elif "child" in q or "children" in q or "family" in q:
        child_means = df.groupby("children")["charges"].mean()
        effect = child_means.max() - child_means.min()
        
        title = f"Average charges range from ${child_means.min():,.0f} (0 children) to ${child_means.max():,.0f}"
        
        fig1 = px.box(df, x="children", y="charges", 
                      title="Medical Charges by Number of Children")
        fig2 = px.bar(x=child_means.index, y=child_means.values,
                      labels={"x": "Number of Children", "y": "Average Charges ($)"},
                      title="Average Charges by Number of Children")
        model = smf.ols("charges ~ children + age + smoker_bin", data=df).fit()

    else:
        return None  # Unsupported question

    # ----- Safer p-value extraction (Bug fix) -----
    # Why? model.params.index[1] can crash if the model structure changes.
    try:
        main_var = model.params.index[1]
        p_value = model.pvalues[main_var]
    except (IndexError, KeyError):
        p_value = model.pvalues.iloc[0]

    return {
        "title": title,
        "effect": effect,
        "p_value": p_value,
        "r_squared": model.rsquared,
        "summary": model.summary().as_text(),
        "fig1": fig1,
        "fig2": fig2
    }

# ============================================================
# 5. Sidebar inputs + buttons
# ============================================================
st.sidebar.header("Ask a question")
user_question = st.sidebar.text_area(
    "What would you like to understand about this data?",
    placeholder="e.g. How does smoking affect medical charges after controlling for age and BMI?"
)

# Button 1 – AI Coach (with error handling – Security + Reliability)
if st.sidebar.button("Propose Analysis Approach"):
    if not user_question.strip():
        st.warning("Please type a question first.")
    else:
        st.subheader("AI Statistical Coach")
        try:
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
        except Exception as e:
            # Why try/except?  
            # If the API key is missing, rate-limited, or Groq is down,
            # the app should fail gracefully instead of crashing.
            st.error(f"Sorry, the AI coach is temporarily unavailable.\n\nError: {e}")

# Button 2 – Run Analysis
if st.sidebar.button("Run Analysis & Show Charts"):
    if not user_question.strip():
        st.warning("Please type a question first.")
    else:
        results = run_analysis(df, user_question)
        
        if results is None:
            st.info("I don't have a ready-made analysis for this question yet. Try asking about smoking, BMI, region, or children.")
        else:
            st.subheader("Analysis Results")
            st.metric(results["title"], f"${results['effect']:,.0f}")
            st.write(f"P-value: {results['p_value']:.4f}")
            st.write(f"R-squared: {results['r_squared']:.3f}")
            
            st.plotly_chart(results["fig1"], use_container_width=True)
            if results["fig2"] is not None:
                st.plotly_chart(results["fig2"], use_container_width=True)
            
            with st.expander("Full model summary"):
                st.text(results["summary"])

# ============================================================
# 6. Main page content (always visible)
# ============================================================
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Basic Info")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.write(df.describe())