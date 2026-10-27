# SmartStats – AI Statistical Coach

SmartStats is an AI-powered statistical coaching tool that helps users think clearly about data analysis.

Instead of just running numbers, it first explains the right statistical approach, lists key assumptions, highlights risks and limitations, and only then performs the analysis and shows charts.

It is designed with an actuarial mindset: careful about uncertainty, model risk, and over-claiming results.

---

## Features

- Natural language questions about the Medical Cost Personal Dataset
- AI coach that proposes the statistical approach with assumptions and risks
- One-click analysis + interactive charts
- Transparent model summaries

---

## Example Questions

- How does smoking affect medical charges after controlling for age and BMI?
- What is the effect of BMI on medical charges?
- Do smokers pay significantly more on average?

---

## Tech Stack

- **Frontend**: Streamlit
- **Data & Stats**: pandas, statsmodels
- **Charts**: Plotly
- **LLM**: Groq (Qwen 3.8 27B)
- **Language**: Python



---

### Small notes on good README formatting

- **Start with a clear one-sentence description** of what the project does.
- Use **short sections** with `##` headings so people can scan quickly.
- Put the most important information near the top (what it is + how to run it).
- Use code blocks (```bash) for commands so they are easy to copy.
- Keep the tone simple and direct — no marketing fluff.
- Always include a “How to Run” section for any public project.
- End with license (MIT is the simplest and most common for student projects).


---

## How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/NanaAdwoaDevTech/SmartStats.git
cd SmartStats



