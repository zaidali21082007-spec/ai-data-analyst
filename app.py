"""
app.py
AI Data Analysis Assistant - Track A (Explorer)
Loads a CSV, analyzes it, answers natural-language questions using the
Groq LLM API, generates a chart, and explains the findings.
"""

import os
import streamlit as st
import pandas as pd

from analysis import load_dataset, get_dataset_summary, compute_basic_stats, answer_question
from visualization import generate_chart

try:
    from groq import Groq
except ImportError:
    Groq = None

st.set_page_config(page_title="AI Data Analysis Assistant", page_icon="📊", layout="wide")

st.title("📊 AI Data Analysis Assistant")
st.caption("Upload a CSV, ask questions, get a chart, and receive an AI-generated explanation — powered by Groq.")

# ---------------------------------------------------------------
# Sidebar: Groq API key + chart options
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""),
        help="Get a free key at https://console.groq.com/keys",
    )
    model_name = st.text_input("Groq model", value="llama-3.3-70b-versatile")
    chart_choice = st.selectbox("Chart type", ["Auto", "Bar", "Pie", "Histogram", "Line", "Scatter"])
    st.divider()
    st.caption("No API key? The app still works using rule-based Q&A and explanations.")

groq_client = None
if api_key_input and Groq is not None:
    try:
        groq_client = Groq(api_key=api_key_input)
    except Exception as e:
        st.sidebar.error(f"Could not initialize Groq client: {e}")

# ---------------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

try:
    df = load_dataset(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.success("Dataset loaded successfully!")

with st.expander("🔍 Dataset Summary", expanded=True):
    summary = get_dataset_summary(df)
    c1, c2 = st.columns(2)
    c1.metric("Rows", summary["rows"])
    c2.metric("Columns", summary["columns"])

    st.write("**Column names:**", ", ".join(summary["column_names"]))

    info_df = pd.DataFrame({
        "Column": summary["column_names"],
        "Data Type": [summary["data_types"][c] for c in summary["column_names"]],
        "Missing Values": [summary["missing_values"][c] for c in summary["column_names"]],
    })
    st.dataframe(info_df, use_container_width=True)

    st.write("**Preview:**")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------------
# Step 2: Analyze the Dataset
# ---------------------------------------------------------------
with st.expander("📈 Automatic Statistics"):
    stats = compute_basic_stats(df)
    st.write(f"**Total records:** {stats['total_records']}")

    if stats["numeric"]:
        st.write("**Numeric column stats:**")
        st.dataframe(pd.DataFrame(stats["numeric"]).T, use_container_width=True)

    if stats["categorical"]:
        st.write("**Categorical column stats:**")
        for col, info in stats["categorical"].items():
            st.write(f"- `{col}`: {info['unique_values']} unique values, top = **{info['top_value']}** ({info['top_count']} occurrences)")

# ---------------------------------------------------------------
# Step 3: Answer Natural Language Questions
# ---------------------------------------------------------------
st.subheader("❓ Ask Questions About the Data")
st.caption("Judges can type up to three fixed questions here (e.g. 'Which product had the highest sales?').")

default_questions = [
    "Which category appears most frequently?",
    "What is the average value of the main numeric column?",
    "",
]

q_cols = st.columns(3)
questions = []
for i, col in enumerate(q_cols):
    with col:
        q = st.text_input(f"Question {i + 1}", value=default_questions[i], key=f"q{i}")
        questions.append(q)

if st.button("Get Answers", type="primary"):
    for i, q in enumerate(questions):
        if q.strip():
            with st.spinner(f"Answering question {i + 1}..."):
                answer = answer_question(df, q, groq_client=groq_client, model=model_name)
            st.markdown(f"**Q{i + 1}: {q}**")
            st.write(answer)

# ---------------------------------------------------------------
# Step 4: Generate Chart + Step 5: Explain the Result
# ---------------------------------------------------------------
st.subheader("📊 Visualization & Explanation")

if st.button("Generate Chart"):
    fig, rule_explanation = generate_chart(df, chart_type=chart_choice)
    if fig is None:
        st.warning(rule_explanation)
    else:
        st.plotly_chart(fig, use_container_width=True)

        os.makedirs("charts", exist_ok=True)
        chart_path = os.path.join("charts", "chart.png")
        try:
            fig.write_image(chart_path)
            with open(chart_path, "rb") as f:
                st.download_button("⬇️ Download chart as PNG", f, file_name="chart.png", mime="image/png")
        except Exception:
            st.caption("(Install `kaleido` to enable PNG export: pip install -U kaleido)")

        # AI explanation (Groq if available, otherwise the rule-based one)
        explanation = rule_explanation
        if groq_client is not None:
            try:
                with st.spinner("Generating AI explanation..."):
                    resp = groq_client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": (
                                f"In one short, simple sentence, explain this data finding to a "
                                f"non-technical audience: {rule_explanation}"
                            ),
                        }],
                        temperature=0.3,
                    )
                    explanation = resp.choices[0].message.content.strip()
            except Exception:
                pass

        st.info(f"**Explanation:** {explanation}")

st.divider()
st.caption("Built for the AI Data Analysis Challenge — Track A (Explorer) · Powered by Groq")
