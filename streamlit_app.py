import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np

st.set_page_config(page_title="SC Power Analysis", layout="wide")

st.title("⚡ SC LS Crusher Power Analysis Powered by AI")
st.write(
    "Upload your power meter CSV file. The app will analyze the data (Power Factor, KW, KVAR, etc.), show key EDA, and you can chat with an AI agent about plant energy insights, anomalies, and recommendations. "
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
            df.rename(columns={col: col.strip().replace(" ", "_").replace("(", "").replace(")", "")}, inplace=True)

        # Convert numeric columns
        numeric_cols = ["KWH", "V_31", "VLL"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip().replace({'-': None, '': None}), errors="coerce")
        if "TIME" in df.columns:
            df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")

        st.subheader("Quick EDA")
        st.write(df.describe())
        pf_below_08 = (df['PF'] < 0.8).mean() * 100 if "PF" in df.columns else None
        if pf_below_08 is not None:
            st.metric("Count of times PF < 0.8 (%)", f"{pf_below_08:.2f}")

        # --- Power Factor Distribution plot
        if "PF" in df.columns:
            st.subheader("Power Factor (PF) Distribution")
            fig, ax = plt.subplots(figsize=(5,3))
            df['PF'].hist(bins=40, ax=ax)
            ax.axvline(0.8, color='red', linestyle='--', label='PF = 0.8')
            ax.set_xlabel("Power Factor")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

        # --- Smaller side-by-side scatter plots
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

        # --- PF Distribution: Startup/High Load vs Idle/Low Load ---
        st.subheader("PF Distribution: Startup/High Load vs Idle/Low Load")
        if "KW" in df.columns and "PF" in df.columns:
            high_kw_thresh = df["KW"].quantile(0.90)
            low_kw_thresh = df["KW"].quantile(0.10)
            startup_high_load = df[df["KW"] >= high_kw_thresh]
            idle_low_load = df[df["KW"] <= low_kw_thresh]

            # Histograms overlayed
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(startup_high_load["PF"], bins=30, alpha=0.6, label='Startup/High Load (Top 10%)', color='blue')
            ax.hist(idle_low_load["PF"], bins=30, alpha=0.6, label='Idle/Low Load (Bottom 10%)', color='gray')
            ax.axvline(0.8, color='red', linestyle='--', label='PF = 0.8')
            ax.set_xlabel("Power Factor")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

            # Key summary stats
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

            # --- Compose summary insights for the AI context ---
            summary_insights = f"""
2. Scatterplots:
- PF vs KW: At both high and low KW, PF can be very low—suggests both startup/idle and maybe partial-load operation cause poor PF.
- PF vs KVAR: There is a stronger pattern—when KVAR is high, PF is always low. (This is expected: high reactive power = poor PF.)

2. Are Startup or Idle Periods Causing the Worst PF?
Startup/High Load (Top 10% KW): Mean PF = {high_mean_pf:.2f} (higher than average, but still below 0.8).
Idle/Low Load (Bottom 10% KW): Mean PF = {low_mean_pf:.2f} (very poor; mostly near zero).

Histogram shows idle periods are responsible for the worst PF, but even at high load, PF rarely exceeds 0.8.

Management Insight: Idle periods (when the crusher is on but not loaded) cause the worst power factor. Even at high load, PF is sub-optimal. Action: Address both idle running and overall plant compensation.

3. Template for PF Correction Recommendations & Cost-Saving Estimation
A. Current Situation
- Average PF: {df['PF'].mean():.2f}
- PF < 0.8: {pf_below_08:.2f}% of the time
- Idle Period PF: {low_mean_pf:.2f} (worst)
- Startup/High Load PF: {high_mean_pf:.2f}
"""
            # Add this to the stats summary so the AI can reference it
            stats_summary["high_load_pf"] = high_mean_pf
            stats_summary["low_load_pf"] = low_mean_pf
            stats_summary["insights"] = summary_insights

        # Summarize for AI: Only share aggregates, not whole CSV (for privacy/token limits)
        stats_summary.update(df.describe().to_dict())
        stats_summary["PF_below_0.8_%"] = pf_below_08

    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Chatbot Section ---
st.divider()
st.header("💬 Power Factor Chatbot")

openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Paste your OpenAI API key.")

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # On initial load, give the agent context about the dataset
        if df is not None:
            initial_system_prompt = (
                "You are a data analyst for a cement plant. "
                "You have just analyzed sensor data from a power meter on a limestone crusher. "
                f"Summary stats for this dataset: {stats_summary}.\n"
                "When answering, use these stats and EDA to help the user understand anomalies, "
                "energy optimization, and power factor correction recommendations."
            )
            st.session_state.messages.append({"role": "system", "content": initial_system_prompt})

    # Option to generate insights at the start of the chat
    if st.button("Generate AI Insights"):
        with st.spinner("Generating insights..."):
            user_prompt = (
                "Based on the EDA and summary statistics, provide insights, highlight patterns or anomalies in power factor, "
                "compare startup/high load vs idle/low load PF, and give recommendations for improvement and cost-saving to management."
            )
            messages = st.session_state.messages + [{"role": "user", "content": user_prompt}]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            ai_insights = response.choices[0].message['content']
            st.session_state.messages.append({"role": "assistant", "content": ai_insights})
            with st.chat_message("assistant"):
                st.markdown(ai_insights)

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue  # Don't display the initial system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field
    if prompt := st.chat_input("Ask about your plant's power factor, anomalies, or recommendations..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # OpenAI response
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
