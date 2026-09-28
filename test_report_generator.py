import os
from groq import Groq

# -----------------------------
# Connect to the API
# -----------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# -----------------------------
# Example structured input (this will later come from your actual prediction + SHAP output)
# -----------------------------
village_name = "Village_3"
predicted_risk = "High"
confidence = 0.81
top_shap_features = ["residual_chlorine_mg_l (low)", "turbidity_ntu (high)", "bacterial_contamination (present)"]
rule_based_recommendation = "URGENT: Chlorinate water supply immediately; distribute water purification tablets."

# -----------------------------
# Build a clear, structured prompt
# -----------------------------
prompt_text = f"""
You are helping generate a short, plain-language report for a rural health officer.

Village: {village_name}
Predicted outbreak risk level: {predicted_risk}
Model confidence: {confidence:.0%}
Key factors driving this prediction: {", ".join(top_shap_features)}
System's rule-based recommendation: {rule_based_recommendation}

Write a short report (3-4 sentences) explaining this village's water-borne disease risk
in plain, non-technical language, and what action should be taken. Do not invent any
numbers or facts not given above.
"""

# -----------------------------
# Call the LLM
# -----------------------------
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    max_tokens=300,
    messages=[
        {"role": "user", "content": prompt_text}
    ]
)

report_text = response.choices[0].message.content
print("GENERATED REPORT:\n")
print(report_text)