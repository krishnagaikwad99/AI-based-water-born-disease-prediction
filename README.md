AquaWatch — AI-Based Early Warning System for Water-Borne Diseases

A prototype web-based AI system that predicts village-level water-borne disease outbreak risk using water quality, environmental, and historical case data — with explainable AI, GIS visualization, and LLM-generated plain-language reports.

This README is written as a complete, interview-ready walkthrough of the project: what was built, why each decision was made, and what to say when asked about it.

1. One-line pitch (say this first if asked "tell me about this project")

"I built an end-to-end ML system that predicts whether a village is at Low, Medium, or High risk of a water-borne disease outbreak, using water quality and environmental data. It includes explainable AI (SHAP), a GIS risk map, short-term forecasting, and an LLM that turns predictions into plain-language reports for health workers — all in a Streamlit dashboard."

2. Problem Statement

Water-borne diseases (cholera, typhoid, diarrhea, hepatitis A) spread more readily under specific, measurable conditions: contaminated water (low chlorine, high turbidity, bacterial presence), poor sanitation, heavy rainfall/flooding, and seasonal patterns (monsoon spikes). The goal was to build a system that takes these signals at village level and outputs an actionable risk classification before an outbreak becomes severe — an "early warning," not just a post-hoc report.

Key constraint I set myself: 100% software, no hardware/IoT sensors. All data starts from CSV/simulated sources, with the pipeline designed so real sensor data could be substituted later without changing the architecture.

3. Why Simulated Data (be ready to defend this — it's the #1 question you'll get)

The honest answer: No single public dataset combines village-level water quality + disease case counts + population + environmental data in one place. Real fragments exist (IDSP/NCDC outbreak reports, CPCB water quality data, WHO stats) but stitching them together is a data-engineering project on its own.

What I did instead: Built a synthetic dataset using a domain-rule generator, not random noise. I encoded actual epidemiological logic into the generation process:

risk_score = 0.35×(turbidity) + 0.30×(1−chlorine) + 0.20×(rainfall) + 0.25×(1−sanitation) + 0.30×(bacterial contamination) + 0.15×(previous week's cases)

This means the model has to learn real, causally-structured relationships, not memorize noise. I clearly labeled this as simulated everywhere in code, comments, and this README — never presented as real medical data.

Dataset shape: 20 villages × 52 weeks = 1,040 rows, 18 original columns. Fixed random seed (42) for reproducibility.

4. Project Architecture
Data (CSV) → Cleaning/Preprocessing → EDA → Feature Engineering
→ ML Model Training → Evaluation → SHAP Explainability → Forecasting
→ GIS Map + Recommendation Engine → Streamlit Dashboard (+ LLM reports)

Deliberately no microservices, no Docker, no REST API layer — this is a script/notebook-driven pipeline feeding a single Streamlit app. Kept simple on purpose: explainable in a viva, easy to debug, appropriate for the problem's actual complexity.

Folder structure
water_disease_ews/
├── .streamlit/config.toml      # UI theme
├── data/raw/                   # original simulated CSV
├── data/processed/             # cleaned → feature-engineered → recommendations
├── models/                     # saved .pkl models
├── notebooks/                  # phase-by-phase development
├── src/generate_dataset.py     # dataset generator
├── app.py                      # Streamlit dashboard
├── village_risk_map.html       # generated GIS map
├── test_pipeline.py            # sanity tests
└── README.md
5. Phase-by-Phase Summary (what I actually did, in order)
Phase 1-2 — Dataset Design & Loading

Decided on a panel structure: one row = one village, one week. Defined 18 columns (water quality, environmental, population, sanitation, historical cases, target). Chose classification (Low/Medium/High) over regression on raw case counts, because case counts were highly skewed — risk categories are both easier to model reliably and more directly actionable for health workers than a raw number.

Phase 3 — Cleaning & Preprocessing

Data was already clean (0 missing values, 0 duplicates — confirmed, not assumed). Still added defensive handling (median-fill for numeric NaNs, duplicate-drop) so the pipeline wouldn't break if real data were substituted later — real data almost always has gaps. Sorted by village + date for correct time-ordering (matters a lot later for forecasting).

Phase 4 — EDA

Validated that engineered relationships actually showed up in the data:

Turbidity ↑ and chlorine ↓ as risk level increased (boxplots)
Visible monsoon-season bump in total cases (seasonality)
Correlation heatmap confirmed turbidity, bacterial_contamination, historical_cases_last_4_weeks had the strongest positive correlation with case count; chlorine and sanitation_index had the strongest negative correlation.

This was a sanity check, not just visualization — it directly informed which features to engineer next.

Phase 5 — Feature Engineering

Combined correlated raw signals into composite scores so the model doesn't have to learn simple combinations on its own:

water_quality_risk_score = turbidity + (1−chlorine) + bacterial contamination
environmental_risk_score = rainfall + temperature-adjusted term
sanitation_gap = 1 − sanitation_index (flipped so "higher = worse," consistent direction across all risk features — this matters a lot for SHAP readability later)
One-hot encoded categorical columns (water_source_type, disease_type)
Target encoded: Low=0, Medium=1, High=2
Phase 6 — Baseline Model Training

Trained Logistic Regression (simple, fast, explainable baseline) and Random Forest (captures non-linear interactions). 80/20 stratified train-test split.

Critical decision — avoided data leakage: case_count was excluded from features, since it's literally what generated the target label. Including it would have given ~100% accuracy that meant nothing (the model would just be reading the answer key).

Phase 7 — Model Evaluation
Model	Accuracy	F1 (weighted)
Logistic Regression	0.673	0.675
Random Forest	0.683	0.680

Went beyond accuracy — used precision/recall/F1 per class and confusion matrices, because for an early-warning system, recall on the "High" class matters most (missing a real outbreak is worse than a false alarm). Random Forest selected as the primary model, though the margin over Logistic Regression was small — an honest finding, not hidden.

Phase 8 — Prediction

Built the inference pipeline: take a new row of readings → apply identical preprocessing/feature engineering as training → predict → return risk label + confidence (predict_proba). Handled the common real-world bug of column mismatch by explicitly enforcing X_train.columns order on any new input.

Phase 9 — SHAP Explainability

Used shap.TreeExplainer (fast, exact for tree models) to explain individual predictions, not just global feature importance. This is different from Phase 6's basic feature importance — SHAP tells you why this specific village, this specific week was flagged, which is what a health officer actually needs. Top drivers of "High risk" consistently matched domain expectations: water_quality_risk_score, historical_cases_last_4_weeks, bacterial_contamination, residual_chlorine_mg_l.

Phase 10 — Forecasting (with an honesty check built in)

Key discipline point: before building any forecasting model, I explicitly tested whether the data supported it, rather than assuming. Built lag features (cases_lag_1/2/3 per village) and a shifted target (future_risk_level = next week's risk). Trained a 1-week-ahead model and — critically — compared it against a naive baseline (assume next week = this week).

Result: naive baseline = 60.3% accuracy, model = 67.7% accuracy. A real, if modest, ~7.4 point improvement — enough to call it genuine forecasting skill, not just noise. This comparison is the single most important rigor-check in the whole project; it's why I didn't just build an LSTM and claim it "worked" — I avoided deep learning on 52 weekly points/village because the data doesn't have enough depth to justify it.

Phase 11 — GIS Risk Map

Used Folium to plot each village as a color-coded marker (green/orange/red) based on its latest predicted risk, with popups showing name/risk/population. Saved as standalone HTML so it works independent of the notebook and embeds cleanly into Streamlit later.

Phase 12 — Recommendation Engine

Deliberately rule-based, not ML-based. Used np.select with ordered, threshold-based conditions (e.g., chlorine < 0.2 → "chlorinate immediately"). This was an intentional design choice: recommendations need to be transparent and traceable to a specific threshold for a health official to trust and act on — an ML-generated recommendation would be a second black box stacked on the first, and much harder to justify or debug.

Phase 13 — Streamlit Dashboard

Built incrementally in steps: (1) data table + filters, (2) embedded GIS map, (3) manual input form with live prediction, (4) live SHAP explanation for that specific prediction. Used st.session_state to solve a real Streamlit gotcha — the entire script reruns on every button click, so prediction results have to be explicitly persisted in session state or they vanish when a second button (like "Generate Report") is clicked. Reorganized into tabs (Predict / Overview / Map) instead of one long scroll for usability, and used native Streamlit theming (.streamlit/config.toml) rather than custom CSS, which is more reliable and doesn't fight the framework's own rendering.

Phase 14 — Agentic AI Report Generation

Integrated Groq API (fast inference, OpenAI-style SDK) to turn structured prediction output (risk level, confidence, top SHAP factors) into a short plain-language report for non-technical readers. Critical prompt-design decision: explicitly instructed the LLM to use only the facts provided and never invent new numbers — this prevents hallucinated medical claims, which would be dangerous in a health-adjacent tool. This is intentionally scoped as "LLM-assisted report generation," not a complex multi-step agent — an appropriately honest scope for what was actually built.

Phase 15 — Testing & Documentation

Manual end-to-end test script (test_pipeline.py) checking: all required files exist, model predicts without error, no data leakage (case_count correctly excluded), target class distribution isn't degenerate (smallest class = 24.4% of data, well above any imbalance red flag). This README + code comments serve as documentation.

6. Tech Stack
Purpose	Tool
Data handling	Pandas, NumPy
ML models	scikit-learn (Logistic Regression, Random Forest)
Explainability	SHAP
Visualization	Matplotlib, Folium (GIS)
Dashboard	Streamlit
LLM reports	Groq API (openai/gpt-oss-20b)
Storage	CSV files, joblib (.pkl models)
7. Key Engineering Decisions I Should Be Ready to Defend
Decision	Why
Simulated data over scraped/fake-real data	No real combined dataset exists; rule-based generation is honest and reproducible
Classification over regression	Raw case counts were skewed; risk categories are more actionable
Dropped case_count from features	Prevents data leakage — it's the source of the target label
Random Forest over deep learning	Small dataset (1,040 rows); RF gives comparable performance with far more explainability
No LSTM for forecasting	Only 52 weekly points/village — not enough time-series depth to justify it; validated lighter approach against a naive baseline instead of overclaiming
Rule-based recommendation engine	Transparency/traceability matters more than sophistication for health guidance
LLM constrained to given facts only	Prevents hallucinated medical claims in a health-adjacent tool
st.session_state for dashboard	Streamlit reruns the whole script per interaction; without this, results vanish on the next button click
8. Results Summary (numbers to have ready)
Classification: Random Forest — 68.3% accuracy, 0.680 weighted F1 (vs. 67.3%/0.675 for Logistic Regression)
Forecasting (1-week-ahead): 67.7% accuracy vs. 60.3% naive baseline
Class balance: High 436 / Medium 350 / Low 254 (no class under 10%, no imbalance-handling needed)
Data leakage check: passed — case_count excluded from all feature sets
9. Honest Limitations (mention these proactively — it reads as maturity, not weakness)
Data is simulated, not from real surveillance systems
Forecasting only validated at 1-week-ahead (2/3/4-week extension scoped but not fully built out)
Recommendation engine is static rule thresholds, not adaptive
No persistent database (SQLite) yet — predictions aren't logged over time
Groq's hosted model catalog changes over time; model name may need updating later
10. Future Work
Swap in real water-quality/disease-surveillance data using the same schema
Extend forecasting to 2/3/4-week horizons with per-horizon naive-baseline comparison
Add SQLite for logging predictions and tracking village trends over time
Add automated alerting for High-risk predictions (e.g., email/SMS to local health authority)
11. How to Run
powershell
python -m pip install -r requirements.txt

# Create a .env file in project root with:
# GROQ_API_KEY=your-key-here

python -m streamlit run app.py

Optional sanity check:

powershell
python test_pipeline.py
12. Anticipated Interview Questions (quick answers)

Q: Why not use a real dataset? No public dataset combines these features at village level; I built a domain-rule-based synthetic one instead of pretending scraped/invented data was real, and designed the pipeline so real data with the same schema could be substituted without changing any modeling code.

Q: How do you know your model isn't overfitting? Checked train vs. test accuracy gap (not degenerate), used stratified splits, and — most importantly — validated forecasting against a naive baseline rather than trusting raw accuracy alone.

Q: What's the difference between your Phase 6 feature importance and Phase 9 SHAP? Feature importance (Random Forest's built-in) is a global summary — averaged across all predictions. SHAP explains individual predictions — this specific village, this specific week — which is what's actually needed to justify a real-world alert to a health worker.

Q: Why is the recommendation engine not ML-based? Deliberate choice — traceable, threshold-based rules are easier to trust, audit, and act on for health guidance than a second black-box model stacked on the first.

Q: What would you do differently with more time? Get real data, extend forecasting horizons, add a persistent database for trend tracking, and formally handle model retraining as new data arrives.