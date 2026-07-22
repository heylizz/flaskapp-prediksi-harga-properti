"""
Aplikasi Web Prediksi Harga Properti Yogyakarta
Final Project - Machine Learning
Model: Random Forest Regressor (skema eksperimen terbaik, lihat notebook)
"""
from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
KABUPATEN_LIST = bundle["kabupaten_list"]
FEATURE_ORDER = bundle["feature_order"]
METRICS = bundle["metrics"]
MODEL_NAME = bundle["model_name"]


def build_feature_vector(kota, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, garasi):
    row = {
        "luas_tanah": luas_tanah,
        "luas_bangunan": luas_bangunan,
        "kamar_tidur": kamar_tidur,
        "kamar_mandi": kamar_mandi,
        "garasi": garasi,
    }
    for k in KABUPATEN_LIST:
        row[f"kab_{k}"] = 1 if k == kota else 0
    return pd.DataFrame([row])[FEATURE_ORDER]


def format_rupiah(value):
    return f"Rp {value:,.0f}".replace(",", ".")


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    error = None
    form_values = {
        "kota": "", "luas_tanah": "", "luas_bangunan": "",
        "kamar_tidur": "", "kamar_mandi": "", "garasi": "",
    }

    if request.method == "POST":
        form_values["kota"] = request.form.get("kota", "")
        form_values["luas_tanah"] = request.form.get("luas_tanah", "")
        form_values["luas_bangunan"] = request.form.get("luas_bangunan", "")
        form_values["kamar_tidur"] = request.form.get("kamar_tidur", "")
        form_values["kamar_mandi"] = request.form.get("kamar_mandi", "")
        form_values["garasi"] = request.form.get("garasi", "")

        try:
            kota = form_values["kota"]
            luas_tanah = float(form_values["luas_tanah"])
            luas_bangunan = float(form_values["luas_bangunan"])
            kamar_tidur = int(form_values["kamar_tidur"])
            kamar_mandi = int(form_values["kamar_mandi"])
            garasi = int(form_values["garasi"])

            if kota not in KABUPATEN_LIST:
                raise ValueError("Kota/kabupaten tidak valid.")
            if luas_tanah <= 0 or luas_bangunan <= 0:
                raise ValueError("Luas tanah dan luas bangunan harus lebih dari 0.")

            X = build_feature_vector(kota, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, garasi)
            pred = model.predict(X)[0]

            hasil = {
                "harga_rupiah": format_rupiah(pred),
                "harga_miliar": f"{pred/1_000_000_000:.3f}",
                "input": {
                    "kota": kota,
                    "luas_tanah": form_values["luas_tanah"],
                    "luas_bangunan": form_values["luas_bangunan"],
                    "kamar_tidur": form_values["kamar_tidur"],
                    "kamar_mandi": form_values["kamar_mandi"],
                    "garasi": form_values["garasi"],
                },
            }
        except (ValueError, TypeError) as e:
            error = f"Input tidak valid: {e}"

    return render_template(
        "index.html",
        kota_list=KABUPATEN_LIST,
        hasil=hasil,
        error=error,
        form_values=form_values,
        best_model_name=f"{MODEL_NAME} (R2={METRICS['r2']:.3f}, MAPE={METRICS['mape']:.1f}%)",
        is_synthetic=False,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
