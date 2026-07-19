"""
WasteWise AI — Backend API
--------------------------
Loads the trained Random Forest model (produced by the Colab notebook) and
exposes a single endpoint, POST /predict, that returns:
  - the predicted Wastage Food Amount
  - an AI-generated (Groq LLM) recommendation, grounded in data-driven triggers

Run locally with:  python app.py
Deploy on Render as a Python Web Service (see README.md in this folder).
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the frontend (hosted on a different domain) to call this API

# ---------------------------------------------------------------------------
# Load the artifacts produced by Section 12 of the notebook.
# These three files MUST sit in the same folder as this app.py.
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "reference_columns.pkl")
RECDATA_PATH = os.path.join(os.path.dirname(__file__), "recommendation_data.pkl")

try:
    rf_model = joblib.load(MODEL_PATH)
    reference_columns = joblib.load(COLUMNS_PATH)
    recommendation_data = joblib.load(RECDATA_PATH)
    print("Model and supporting data loaded successfully.")
except Exception as e:
    # We don't crash the whole app at import time — instead we surface a clear
    # error on every request, which is much easier to debug during deployment.
    rf_model = None
    reference_columns = None
    recommendation_data = None
    print(f"WARNING: could not load model artifacts: {e}")

event_waste = (recommendation_data or {}).get("event_waste", {})
storage_waste = (recommendation_data or {}).get("storage_waste", {})
season_waste = (recommendation_data or {}).get("season_waste", {})
overall_mean_waste = (recommendation_data or {}).get("overall_mean_waste", 0)

# ---------------------------------------------------------------------------
# Groq LLM setup (used for the recommendation text)
# Set your key as an environment variable named GROQ_API_KEY on Render —
# never hard-code it in this file.
# ---------------------------------------------------------------------------
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

llm = None
if os.environ.get("GROQ_API_KEY"):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

RECOMMENDATION_SYSTEM_PROMPT = """You are a food service sustainability advisor for WasteWise AI.
You will be given: (1) a predicted food wastage amount for an event, and (2) a list of
data-driven observations about that event. Using ONLY the observations provided (do not
invent new facts or numbers), write 2-4 short, practical, business-oriented recommendations
to help reduce food wastage for this specific event. Be concise and actionable. Do not repeat
the raw numbers verbatim as your main point -- turn them into clear advice.
"""


def get_data_triggers(new_event: dict) -> list:
    """Same logic as Section 10 of the notebook — factual, data-backed observations."""
    triggers = []

    e_avg = event_waste.get(new_event.get("Event Type"), overall_mean_waste)
    if e_avg > overall_mean_waste:
        triggers.append(
            f"'{new_event.get('Event Type')}' events average ${e_avg:.2f} wastage, "
            f"above the overall average of ${overall_mean_waste:.2f}."
        )

    s_avg = storage_waste.get(new_event.get("Storage Conditions"), overall_mean_waste)
    if s_avg > overall_mean_waste:
        triggers.append(
            f"'{new_event.get('Storage Conditions')}' storage shows ${s_avg:.2f} average wastage, "
            f"higher than the dataset average."
        )

    se_avg = season_waste.get(new_event.get("Seasonality"), overall_mean_waste)
    if se_avg > overall_mean_waste:
        triggers.append(f"Events during '{new_event.get('Seasonality')}' average ${se_avg:.2f} wastage.")

    if new_event.get("Preparation Method") == "Buffet":
        triggers.append("Buffet-style preparation is generally associated with over-preparation risk.")

    if not triggers:
        triggers.append("No major risk factors detected — this event profile aligns with typical low-wastage patterns.")

    return triggers


def generate_recommendation(new_event: dict, predicted_wastage: float) -> str:
    triggers = get_data_triggers(new_event)

    if llm is None:
        # Graceful fallback if no Groq key is configured — still useful, just not LLM-written.
        return "Recommendations based on data patterns:\n" + "\n".join(f"- {t}" for t in triggers)

    triggers_text = "\n".join(f"- {t}" for t in triggers)
    user_message = (
        f"Predicted Wastage Food Amount: ${predicted_wastage}\n"
        f"Event details: {new_event}\n\n"
        f"Data-driven observations:\n{triggers_text}"
    )
    response = llm.invoke([
        SystemMessage(content=RECOMMENDATION_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    return response.content.strip()


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "WasteWise AI backend is running."})


@app.route("/predict", methods=["POST"])
def predict():
    if rf_model is None:
        return jsonify({"error": "Model not loaded on server. Check deployment logs."}), 500

    payload = request.get_json(force=True)

    required_fields = [
        "Type of Food", "Number of Guests", "Event Type", "Quantity of Food",
        "Storage Conditions", "Purchase History", "Seasonality",
        "Preparation Method", "Geographical Location", "Pricing",
    ]
    missing = [f for f in required_fields if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Encode the same way the notebook did: one-hot encode, then align columns
    new_df = pd.DataFrame([payload])
    new_encoded = pd.get_dummies(new_df)
    new_encoded = new_encoded.reindex(columns=reference_columns, fill_value=0)

    prediction = float(rf_model.predict(new_encoded)[0])
    prediction = round(prediction, 2)

    recommendation = generate_recommendation(payload, prediction)

    return jsonify({
        "predicted_wastage": prediction,
        "recommendation": recommendation,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
