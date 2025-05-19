import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np

st.set_page_config(page_title="SC Power Analysis", layout="wide")

st.title("⚡ SC LS Crusher Power Analysis Powered by AI")
st.write(
    "Upload your power meter CSV file. The app will analyze the data (Power Factor, KW, KVAR, etc.), "
    "show key EDA, and you can chat with an AI agent about plant energy insights, anomalies, and recommendations. "
    "You need your OpenAI API key to use the chatbot."
)

# --- Data Upload & EDA Section ---
uploaded_file = st.file_uploader("Upload Power Meter CSV", type=["csv"])
df = None
summary_insights = ""
stats_summary = {}

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded! Preview below.")
        st.dataframe(df.head())

        # Basic column cleanup
        for col in df.columns:
            df.rename(
                columns={col: col.strip().replace(" ", "_").replace("(", "").replace(")", "")},
                inplace=True
            )

        # Convert numeric columns
        numeric_cols = ["KWH", "V_31", "VLL"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str)
                        .str.replace(",", "")
                        .str.strip()
                        .replace({'-': None, '': None}),
                    errors="coerce"
                )
        if "TIME" in df.columns:
            df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")

        st.subheader("Quick EDA")
        st.write(df.describe())
        pf_below_08 = (df['PF'] < 0.8).mean() * 100 if "PF" in df.columns else None
        if pf_below_08 is not None:
            st.metric("Count of times PF < 0.8 (%)", f"{pf_below_08:.2f}")

        # --- Power Factor Distribution plot ---
        if "PF" in df.columns:
            st.subheader("Power Factor (PF) Distribution")
            fig, ax = plt.subplots(figsize=(5,3))
            df['PF'].hist(bins=40, ax=ax)
            ax.axvline(0.8, color='red', linestyle='--', label='PF = 0.8')
            ax.set_xlabel("Power Factor")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

        # --- Scatter Plots ---
        st.subheader("Scatter Plots: PF vs KW, PF vs KVAR")
        col1, col2 = st.columns(2)
        if "KW" in df.columns and "PF" in df.columns:
            with col1:
                fig, ax = plt.subplots(figsize=(4,3))
                ax.scatter(df["KW"], df["PF"], alpha=0.3, s=5)
                ax.set_xlabel("KW")
                ax.set_ylabel("PF")
                ax.set_title("PF vs KW")
                st.pyplot(fig)
        if "KVAR" in df.columns and "PF" in df.columns:
            with col2:
                fig, ax = plt.subplots(figsize=(4,3))
                ax.scatter(df["KVAR"], df["PF"], alpha=0.3, s=5, color='orange')
                ax.set_xlabel("KVAR")
                ax.set_ylabel("PF")
                ax.set_title("PF vs KVAR")
                st.pyplot(fig)

        # --- PF Distribution: High vs Low Load ---
        st.subheader("PF Distribution: Startup/High Load vs Idle/Low Load")
        if "KW" in df.columns and "PF" in df.columns:
            high_kw_thresh = df["KW"].quantile(0.90)
            low_kw_thresh = df["KW"].quantile(0.10)
            startup_high_load = df[df["KW"] >= high_kw_thresh]
            idle_low_load = df[df["KW"] <= low_kw_thresh]

            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(startup_high_load["PF"], bins=30, alpha=0.6, label='Startup/High Load (Top 10%)', color='blue')
            ax.hist(idle_low_load["PF"], bins=30, alpha=0.6, label='Idle/Low Load (Bottom 10%)', color='gray')
            ax.axvline(0.8, color='red', linestyle='--', label='PF = 0.8')
            ax.set_xlabel("Power Factor")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

            high_mean_pf = startup_high_load["PF"].mean()
            low_mean_pf = idle_low_load["PF"].mean()
            high_pf_count = startup_high_load["PF"].count()
            low_pf_count = idle_low_load["PF"].count()

            st.markdown(f"""
            **Startup/High Load (Top 10% KW):**  
            - Mean PF: `{high_mean_pf:.2f}`  
            - Number of Records: `{high_pf_count}`  

            **Idle/Low Load (Bottom 10% KW):**  
            - Mean PF: `{low_mean_pf:.2f}`  
            - Number of Records: `{low_pf_count}`  
            """)

            # Compose narrative insights
            summary_insights = f"""
2. Scatterplots:
- PF vs KW: At both high and low KW, PF can be very low—suggests both startup/idle and maybe partial-load operation cause poor PF.
- PF vs KVAR: There is a stronger pattern—when KVAR is high, PF is always low.

3. PF by Load:
- Startup/High Load (Top 10% KW): Mean PF = {high_mean_pf:.2f}
- Idle/Low Load (Bottom 10% KW): Mean PF = {low_mean_pf:.2f}

Management Insight: Idle periods cause the worst PF. Even at high load, PF rarely exceeds 0.8. Action: Address both idle running and plant compensation.
"""

            stats_summary["high_load_pf"] = high_mean_pf
            stats_summary["low_load_pf"]  = low_mean_pf
            stats_summary["insights"]     = summary_insights

        stats_summary.update(df.describe().to_dict())
        stats_summary["PF_below_0.8_%"] = pf_below_08

        # ——— Convert stats_summary to a simple Markdown table ———
        md_lines = ["| Metric | Value |", "|---|---|"]
        for metric, value in stats_summary.items():
            md_lines.append(f"| {metric} | {value} |")
        st.session_state["stats_md"] = "\n".join(md_lines)

    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Chatbot Section ---
st.divider()
st.header("💬 Chat to find Insights")

# grab your key however you like
openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

if "ai_insights" not in st.session_state:
    st.session_state["ai_insights"] = ""
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Generate AI Insights button
st.markdown("---")
if st.button("Generate AI Insights"):
    with st.spinner("Generating summary insights…"):
        ai_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data-savvy energy analyst. "
                        "Given the summary statistics and chart descriptions above, "
                        "provide a concise executive summary highlighting key findings, "
                        "anomalies, and high-level recommendations."
                    )
                },
                {"role": "user", "content": summary_insights}
            ]
        )
    st.session_state["ai_insights"] = ai_resp.choices[0].message.content

# Display AI Insights
if st.session_state["ai_insights"]:
    st.subheader("🤖 AI-Generated Insights")
    st.info(st.session_state["ai_insights"])

# Chat input
st.markdown("---")
st.subheader("💬 Ask AI About Your Energy Data")
user_q = st.chat_input("Type your question here…")
if user_q:
    st.session_state["chat_history"].append({"role": "user", "content": user_q})

    # Build system context with both AI insights and raw stats table
    system_context = (
        "You are a helpful assistant for plant energy analysis. "
        "Use the following AI‐generated insights and raw summary stats to ground your answers:\n\n"
        f"**AI Insights:**\n{st.session_state['ai_insights']}\n\n"
        f"**Raw Stats Summary:**\n{st.session_state['stats_md']}"
    )

    messages = [
        {"role": "system", "content": system_context},
        *st.session_state["chat_history"]
    ]

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    ).choices[0].message

    st.session_state["chat_history"].append({"role": reply.role, "content": reply.content})

# Render chat history
for msg in st.session_state["chat_history"]:
    st.chat_message(msg["role"]).write(msg["content"])
