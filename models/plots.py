# models/plots.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay


# Diretório padrão de saída dos gráficos
PLOTS_DIR = "web/static/plots"


def ensure_dir():
    """Garante que o diretório de plots exista."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


# 1️⃣ Decision Boundary (Random Forest)
def plot_decision_boundary(model, X, y, x_col, y_col):
    ensure_dir()
    path = os.path.join(PLOTS_DIR, "decision_boundary.png")

    fig, ax = plt.subplots(figsize=(6, 5))

    try:
        DecisionBoundaryDisplay.from_estimator(
            model,
            X,
            response_method="predict",
            cmap=plt.cm.RdYlBu,
            alpha=0.4,
            ax=ax
        )
    except Exception:
        pass

    scatter = ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=y,
        cmap=plt.cm.RdYlBu,
        edgecolor="k",
        s=40,
        alpha=0.7
    )

    ax.set_title("Random Forest - Decision Boundary")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.colorbar(scatter, ax=ax, label="Aceitação")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# 2️⃣ Matriz de Confusão
def plot_confusion_matrix(cm):
    ensure_dir()
    path = os.path.join(PLOTS_DIR, "confusion_rf.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    ax.set_title("Matriz de Confusão - Random Forest")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# 3️⃣ Feature Importance
def plot_feature_importance(importances, features):
    ensure_dir()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(
        features,
        importances,
        color=["#2ecc71", "#3498db"],
        edgecolor="black"
    )

    ax.set_title("Importância das Features - RF")
    ax.set_xlabel("Importância")

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{importances[i]:.3f}",
            va="center",
            ha="left",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# 4️⃣ Distribuição 3D (opcional)
def plot_3d_distribution(df, x_col, y_col, z_col, y):
    if z_col == "Qualquer":
        return

    ensure_dir()
    path = os.path.join(PLOTS_DIR, "distribution_3d.png")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=y,
        cmap=plt.cm.RdYlBu,
        edgecolor="k",
        alpha=0.6,
        s=30
    )

    ax.set_title("Distribuição no Espaço 3D")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    plt.colorbar(scatter, ax=ax, label="Aceitação")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# 5️⃣ Comparação de Modelos
def plot_model_comparison(results):
    ensure_dir()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")

    models = ["Random Forest", "Extra Trees"]
    accuracies = [
        results["rf"]["test_accuracy"],
        results["et"]["test_accuracy"]
    ]

    if results.get("optimized"):
        models.append("RF Otimizado")
        accuracies.append(results["optimized"]["test_accuracy"])

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        models,
        accuracies,
        color=["#3498db", "#e74c3c", "#2ecc71"][:len(models)]
    )

    ax.set_title("Comparação de Acurácia")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# 6️⃣ Distribuição de Probabilidades
def plot_probability_distribution(model, X, y):
    ensure_dir()
    path = os.path.join(PLOTS_DIR, "probability_distribution.png")

    y_proba = model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        [y_proba[y == 0], y_proba[y == 1]],
        bins=20,
        label=["Rejeitou", "Aceitou"],
        color=["#e74c3c", "#2ecc71"],
        alpha=0.7,
        edgecolor="black"
    )

    ax.set_title("Distribuição de Probabilidades - RF")
    ax.set_xlabel("Probabilidade de Aceitação")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.axvline(0.5, linestyle="--", color="gray", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
