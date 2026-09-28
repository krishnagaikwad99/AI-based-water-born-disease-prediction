import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AquaWatch - Disease Early Warning",
    page_icon="💧",
    layout="wide"
)

st.title("💧 AquaWatch")
st.caption("AI-Based Early Warning System for Water-Borne Diseases  ·  Prototype using simulated data")

script_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load data and model
# -----------------------------
data = pd.read_csv(os.path.join(script_dir, "data", "processed", "village_recommendations.csv"))
rf_model = joblib.load(os.path.join(script_dir, "models", "random_forest_model.pkl"))

training_columns_df = pd.read_csv(os.path.join(script_dir, "data", "processed", "features_disease_dataset.csv"))
drop_cols_for_features = ["village_id", "village_name", "date", "outbreak_risk_level",
                           "outbreak_risk_level_encoded", "case_count"]
model_columns = training_columns_df.drop(columns=drop_cols_for_features).columns

explainer = shap.TreeExplainer(rf_model)

# -----------------------------
# Tabs = the main navigation
# -----------------------------
tab_predict, tab_overview, tab_map = st.tabs(["🔎 Predict Risk", "🏘️ Village Overview", "🗺️ Risk Map"])

# =========================================================
# TAB 1: Manual Input Form + Live Risk Prediction
# =========================================================
with tab_predict:
    st.subheader("Check Risk for New Readings")
    st.caption("Enter this week's readings for a village to get a live risk prediction")

    with st.container(border=True):
        with st.form("prediction_form"):
            input_col1, input_col2, input_col3 = st.columns(3)

            with input_col1:
                ph_level = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
                turbidity_ntu = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.0, step=0.1)
                residual_chlorine_mg_l = st.number_input("Residual Chlorine (mg/L)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

            with input_col2:
                rainfall_mm = st.number_input("Rainfall (mm, past week)", min_value=0.0, value=10.0, step=1.0)
                temperature_c = st.number_input("Temperature (C)", min_value=0.0, max_value=50.0, value=28.0, step=0.5)
                sanitation_index = st.slider("Sanitation Index (0=poor, 1=excellent)", 0.0, 1.0, 0.6)

            with input_col3:
                bacterial_contamination = st.selectbox("Bacterial Contamination Detected?", ["No", "Yes"])
                historical_cases_last_4_weeks = st.number_input("Cases in Last 4 Weeks", min_value=0, value=2, step=1)
                population = st.number_input("Village Population", min_value=1, value=5000, step=100)

            water_source_type = st.selectbox("Water Source Type", ["borewell", "handpump", "pond", "river", "municipal_supply"])
            disease_type = st.selectbox("Disease to Assess", ["cholera", "typhoid", "diarrhea", "hepatitis_a"])
            reading_date = st.date_input("Reading Date")

            submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        bacterial_flag = 1 if bacterial_contamination == "Yes" else 0
        water_quality_risk_score = (turbidity_ntu / 10) + (1 - residual_chlorine_mg_l) + bacterial_flag
        environmental_risk_score = (rainfall_mm / 50) + ((temperature_c - 20) / 15)
        sanitation_gap = 1 - sanitation_index
        week_number = reading_date.isocalendar()[1]
        month = reading_date.month

        input_dict = {
            "latitude": 16.7, "longitude": 74.2,
            "population": population,
            "ph_level": ph_level,
            "turbidity_ntu": turbidity_ntu,
            "residual_chlorine_mg_l": residual_chlorine_mg_l,
            "bacterial_contamination": bacterial_flag,
            "rainfall_mm": rainfall_mm,
            "temperature_c": temperature_c,
            "sanitation_index": sanitation_index,
            "historical_cases_last_4_weeks": historical_cases_last_4_weeks,
            "water_quality_risk_score": water_quality_risk_score,
            "environmental_risk_score": environmental_risk_score,
            "sanitation_gap": sanitation_gap,
            "week_number": week_number,
            "month": month,
        }

        for col in model_columns:
            if col.startswith("water_source_type_"):
                source_name = col.replace("water_source_type_", "")
                input_dict[col] = 1 if water_source_type == source_name else 0
            elif col.startswith("disease_type_"):
                disease_name = col.replace("disease_type_", "")
                input_dict[col] = 1 if disease_type == disease_name else 0

        input_row = pd.DataFrame([input_dict])
        input_row = input_row[model_columns]

        prediction = rf_model.predict(input_row)[0]
        probabilities = rf_model.predict_proba(input_row)[0]

        risk_mapping_reverse = {0: "Low", 1: "Medium", 2: "High"}
        predicted_label = risk_mapping_reverse[prediction]
        confidence = probabilities.max()

        shap_values_input = explainer.shap_values(input_row)
        contributions = shap_values_input[0, :, prediction]
        explanation_df = pd.DataFrame({
            "feature": model_columns,
            "impact": contributions
        })
        explanation_df["abs_impact"] = explanation_df["impact"].abs()
        explanation_df = explanation_df.sort_values(by="abs_impact", ascending=False).head(8)

        st.session_state["predicted_label"] = predicted_label
        st.session_state["confidence"] = confidence
        st.session_state["probabilities"] = probabilities
        st.session_state["explanation_df"] = explanation_df
        st.session_state["has_prediction"] = True

    if st.session_state.get("has_prediction", False):
        predicted_label = st.session_state["predicted_label"]
        confidence = st.session_state["confidence"]
        probabilities = st.session_state["probabilities"]
        explanation_df = st.session_state["explanation_df"]

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            if predicted_label == "High":
                st.error(f"**Risk Level: {predicted_label}**\n\nConfidence: {confidence:.0%}", icon="🔴")
            elif predicted_label == "Medium":
                st.warning(f"**Risk Level: {predicted_label}**\n\nConfidence: {confidence:.0%}", icon="🟠")
            else:
                st.success(f"**Risk Level: {predicted_label}**\n\nConfidence: {confidence:.0%}", icon="🟢")

        with result_col2:
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            prob_col1.metric("Low", f"{probabilities[0]:.0%}")
            prob_col2.metric("Medium", f"{probabilities[1]:.0%}")
            prob_col3.metric("High", f"{probabilities[2]:.0%}")

        with st.expander("🧠 Why this prediction? (SHAP explanation)", expanded=True):
            fig, ax = plt.subplots(figsize=(7, 3.5))
            colors = ["#dc2626" if val > 0 else "#059669" for val in explanation_df["impact"]]
            ax.barh(explanation_df["feature"], explanation_df["impact"], color=colors)
            ax.set_xlabel(f"Impact on '{predicted_label}' prediction")
            ax.invert_yaxis()
            st.pyplot(fig)
            st.caption("Red bars push toward this risk level  ·  Green bars push away from it")

        with st.expander("📝 Plain-language report", expanded=False):
            if st.button("Generate Report", use_container_width=True):
                from groq import Groq

                groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

                top_features_text = [
                    f"{row['feature']} (impact: {row['impact']:.2f})"
                    for _, row in explanation_df.head(3).iterrows()
                ]

                report_prompt = f"""
You are helping generate a short, plain-language report for a rural health officer.

Village readings entered manually for assessment.
Predicted outbreak risk level: {predicted_label}
Model confidence: {confidence:.0%}
Key factors driving this prediction: {", ".join(top_features_text)}

Write a short report (3-4 sentences) explaining this risk level in plain,
non-technical language, and what action should be taken. Do not invent any
numbers or facts not given above.
"""
                with st.spinner("Generating report..."):
                    groq_response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        max_tokens=300,
                        messages=[{"role": "user", "content": report_prompt}]
                    )
                    generated_report = groq_response.choices[0].message.content

                st.info(generated_report)

# =========================================================
# TAB 2: Village Risk Overview
# =========================================================
with tab_overview:
    st.subheader("Village Risk Overview")

    risk_filter = st.multiselect(
        "Filter by risk level",
        options=data["predicted_risk"].unique(),
        default=list(data["predicted_risk"].unique())
    )

    filtered_data = data[data["predicted_risk"].isin(risk_filter)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Villages Shown", len(filtered_data))
    col2.metric("High Risk Villages", (filtered_data["predicted_risk"] == "High").sum())
    col3.metric("Total Population Covered", int(filtered_data["population"].sum()))

    st.dataframe(filtered_data, use_container_width=True, hide_index=True)

# =========================================================
# TAB 3: Village Risk Map
# =========================================================
with tab_map:
    st.subheader("Village Risk Map")
    st.caption("🟢 Low risk   🟠 Medium risk   🔴 High risk")

    map_path = os.path.join(script_dir, "village_risk_map.html")
    with open(map_path, "r", encoding="utf-8") as map_file:
        map_html = map_file.read()

    components.html(map_html, height=550, scrolling=False)