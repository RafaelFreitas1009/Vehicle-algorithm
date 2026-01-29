# app.py
from __future__ import annotations

from flask import Flask, render_template, request, jsonify

from models.explainer import generate_business_explanation
from data.processor import DataProcessor
from models.coupon_model import CouponModel
from utils.validators import normalize_axes, ensure_min_samples

from models.plots import (
    plot_decision_boundary,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_3d_distribution,
    plot_model_comparison,
    plot_probability_distribution
)

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)

# =========================
# ESTADO EM MEMÓRIA
# =========================
LAST_RUN = {
    "results": None,
    "messages": [],
}

# =========================
# DADOS (processados 1x)
# =========================
processor = DataProcessor("assets/in-vehicle-coupon-recommendation.csv")
processor.process_data()


def build_options():
    return {
        "CoffeeHouse": sorted(processor.get_unique_values("CoffeeHouse").astype(str)),
        "destination": sorted(processor.get_unique_values("destination").astype(str)),
        "axes": ["temperature", "age", "income", "CoffeeHouse", "Bar", "RestaurantLessThan20"],
    }


# ======================================================
# CORE DA PIPELINE (reutilizável por HTML e API)
# ======================================================
def execute_pipeline(form_data: dict):
    coffee = form_data.get("coffee", "Qualquer")
    dest = form_data.get("dest", "Qualquer")

    x = form_data.get("x_axis", "temperature")
    y_axis = form_data.get("y_axis", "age")
    z = form_data.get("z_axis", "income")

    optimize = form_data.get("optimize") in ("on", True, "true")

    # --- filtros ---
    d = processor.df_num.copy()
    df_original = processor.df.copy()

    if coffee != "Qualquer":
        d = d[df_original["CoffeeHouse"].astype(str) == coffee]
    if dest != "Qualquer":
        d = d[df_original["destination"].astype(str) == dest]

    ensure_min_samples(len(d), minimum=50)

    # --- eixos ---
    x_col, y_col, z_col, changed, msgs = normalize_axes(x, y_axis, z)

    X = d[[x_col, y_col]]
    y_true = d["Y"]

    # --- treino ---
    model = CouponModel()
    model.prepare_train_test(X, y_true)

    rf_results = model.train_random_forest()
    et_results = model.train_extra_trees()
    opt_results = model.optimize_hyperparameters() if optimize else None

    # --- plots ---
    plot_decision_boundary(model.rf_model, X, y_true, x_col, y_col)
    plot_confusion_matrix(model.results["rf"]["confusion_matrix"])
    plot_feature_importance(
        model.results["rf"]["feature_importance"],
        [x_col, y_col]
    )
    plot_probability_distribution(model.rf_model, X, y_true)
    plot_3d_distribution(d, x_col, y_col, z_col, y_true)
    plot_model_comparison(model.results)

    results = {
        "rf": rf_results,
        "et": et_results,
        "optimized": opt_results,
        "meta": {
            "n_samples": len(d),
            "x_axis": x_col,
            "y_axis": y_col,
            "z_axis": z_col,
            "filters": {"coffee": coffee, "dest": dest},
        },
    }

    return results, msgs if changed else []


# =========================
# ROTAS
# =========================
@app.get("/")
def index():
    return render_template("index.html", options=build_options(), last=LAST_RUN)


# ---------- HTML (fallback / tradicional) ----------
@app.post("/run")
def run():
    results, messages = execute_pipeline(request.form)

    LAST_RUN["results"] = results
    # 🧠 Geração do texto inteligente
    LAST_RUN["explanation"] = generate_business_explanation(LAST_RUN["results"])
    LAST_RUN["messages"] = messages

    return render_template(
        "index.html",
        options=build_options(),
        last=LAST_RUN
    )


# ---------- API (JavaScript / fetch) ----------
@app.post("/api/run")
def run_api():
    try:
        data = request.get_json(force=True)
        results, messages = execute_pipeline(data)

        LAST_RUN["results"] = results
        LAST_RUN["messages"] = messages

        return jsonify({
            "status": "ok",
            "meta": results["meta"],
            "metrics": {
                "rf_accuracy": results["rf"]["test_accuracy"],
                "et_accuracy": results["et"]["test_accuracy"],
                "optimized_accuracy": (
                    results["optimized"]["test_accuracy"]
                    if results["optimized"] else None
                ),
            },
            "messages": messages,
            "plots": {
                "decision_boundary": "decision_boundary.png",
                "confusion": "confusion_rf.png",
                "feature_importance": "feature_importance.png",
                "probability": "probability_distribution.png",
                "model_comparison": "model_comparison.png",
                "distribution_3d": "distribution_3d.png",
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


def run_app():
    app.run(debug=True)
