import os
import math
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from local_transformer_intent import LocalIntentPipeline
from recommendation import RecommendationEngine
from package_planner import WeddingPackagePlanner

# optional seq2seq
try:
    from seq2seq_generator import generate_reply_safe
    USE_SEQ2SEQ = True
except:
    USE_SEQ2SEQ = False


app = Flask(__name__)
CORS(app)

# =========================
# GLOBAL STATE
# =========================
ai_pipeline = None
db = None
recommendation_engine = None
package_planner = None
wedding_dataset = None


# =========================
# SAFE JSON SANITIZER
# =========================
def _sanitize_for_json(obj):
    import pandas as _pd
    import numpy as np
    try:
        isna = _pd.isna
    except:
        isna = lambda x: False

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]

    if isinstance(obj, (np.floating, np.integer)):
        try:
            return obj.item()
        except:
            return float(obj)

    try:
        if isna(obj):
            return None
    except:
        pass

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj


# =========================
# DB FALLBACK
# =========================
def get_db_connection():
    return None


# =========================
# INIT SYSTEM
# =========================
def initialize_system():
    global ai_pipeline, db, recommendation_engine, package_planner, wedding_dataset

    print("🚀 Initializing Wedding System...")

    wedding_dataset = pd.read_csv("dataset_pertanyaan_wedding.csv")

    if "harga_min" in wedding_dataset.columns:
        wedding_dataset["harga_min"] = wedding_dataset["harga_min"].fillna(0).astype(int)

    if "harga_max" in wedding_dataset.columns:
        wedding_dataset["harga_max"] = wedding_dataset["harga_max"].fillna(999999999).astype(int)

    print("✓ CSV Loaded")

    if ai_pipeline is None:
        ai_pipeline = LocalIntentPipeline("models/local_transformer_intent")
        print("✓ Local Intent Model Ready")

    db_conn = get_db_connection()

    if not db_conn:
        print("⚠️ DB Not Connected → Using CSV Fallback")
        db = wedding_dataset
    else:
        db = db_conn
        print("✓ Database Ready")

    recommendation_engine = RecommendationEngine(n_clusters=3)
    print("✓ Recommendation Engine Ready")

    package_planner = WeddingPackagePlanner(db)
    print("✓ Package Planner Ready")

    print("🎉 System Ready!")


# =========================
# REPLY GENERATOR
# =========================
def generate_assistant_reply(ai_result: dict, clustering_result: dict, items: list):
    intent = ai_result.get('intent_pred', '')
    slots = ai_result.get('slots', {})

    if intent == 'estimasi_budget':
        reply = "Kamu ingin estimasi budget ya."
    elif intent in ['cari_venue', 'cari_dekor', 'cari_catering', 'cari_vendor']:
        reply = "Baik, aku carikan rekomendasinya ya."
    elif intent == 'tanya_kemungkinan':
        reply = "Oke, aku cek kemungkinan opsinya dulu."
    else:
        reply = "Berikut rekomendasi terbaik untuk kamu."

    parts = [reply]

    if slots:
        slot_text = []
        if slots.get('tema'):
            slot_text.append(f"tema {slots['tema']}")
        if slots.get('lokasi'):
            slot_text.append(f"lokasi {slots['lokasi']}")
        if slots.get('budget_min') or slots.get('budget_max'):
            bmin = slots.get('budget_min')
            bmax = slots.get('budget_max')
            if bmin and bmax:
                slot_text.append(f"budget Rp {int(bmin):,} - Rp {int(bmax):,}")
            elif bmin:
                slot_text.append(f"budget minimal Rp {int(bmin):,}")
            elif bmax:
                slot_text.append(f"budget maksimal Rp {int(bmax):,}")

        if slot_text:
            parts.append("Dengan preferensi: " + ", ".join(slot_text))

    if clustering_result and clustering_result.get("recommendations"):
        parts.append(
            f"Aku temukan {len(clustering_result['recommendations'])} pilihan cocok 😊"
        )

    return " ".join(parts)


# =========================
# API
# =========================
@app.route("/api/process", methods=["POST"])
def chat_api():
    global ai_pipeline, recommendation_engine, package_planner, wedding_dataset

    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "text kosong"}), 400

    ai_result = ai_pipeline(text)

    clustering_result = recommendation_engine.recommend(
        dataset=wedding_dataset,
        user_slots=ai_result.get("slots", {})
    )

    if package_planner:
        package = package_planner.plan(ai_result.get("slots", {}))
    else:
        package = None

    try:
        if USE_SEQ2SEQ:
            assistant_reply = generate_reply_safe(text)
        else:
            assistant_reply = generate_assistant_reply(ai_result, clustering_result, [])
    except:
        assistant_reply = generate_assistant_reply(ai_result, clustering_result, [])

    response = {
        "user_text": text,
        "intent": ai_result.get("intent_pred"),
        "slots": ai_result.get("slots"),
        "probabilities": ai_result.get("probs"),
        "recommendations": clustering_result,
        "wedding_package": package,
        "assistant_reply": assistant_reply
    }

    return jsonify(_sanitize_for_json(response))


# =========================
# RUN
# =========================
initialize_system()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


def get_csv_recommendations(intent, slots):
    df = wedding_dataset.copy()

    # intent2 yang dianggap rekomendasi
    if intent not in [
        "cari_rekomendasi_paket",
        "cari_vendor",
        "cari_venue",
        "cari_catering"
    ]:
        return []

    # filter lokasi
    if slots.get("lokasi"):
        df = df[df["lokasi"].str.contains(slots["lokasi"], case=False, na=False)]

    # filter budget
    if slots.get("budget_min"):
        budget = slots["budget_min"]
        df = df[
            (df["harga_min"] <= budget) &
            (df["harga_max"] >= budget)
        ]

    if len(df) == 0:
        return []

    # ambil top 5
    results = df.head(5)

    rekom = []
    for _, row in results.iterrows():
        rekom.append({
            "nama": row.get("nama", "-"),
            "lokasi": row.get("lokasi", "-"),
            "harga": f"{row.get('harga_min',0)} - {row.get('harga_max',0)}",
            "kategori": row.get("kategori","-")
        })

    return rekom
