"""
Generate plots for CNN regression results.
Reads from results/cnn_regression/ and saves PNGs to the same directory.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "cnn_regression"


def plot_loss_curve(history: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_mse"], "o-", label="Train MSE", color="#2563eb")
    ax.plot(history["epoch"], history["val_mse"], "s-", label="Val MSE", color="#dc2626")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Squared Error", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(history["epoch"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_actual_vs_predicted(preds: pd.DataFrame, metrics: dict, out: Path):
    actual = preds["actual"].values
    predicted = preds["predicted"].values

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.5, s=30, color="#6366f1", edgecolors="white", linewidths=0.3)

    lo, hi = min(actual.min(), predicted.min()) - 0.2, max(actual.max(), predicted.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "--", color="#9ca3af", linewidth=1.5, label="Perfect prediction")

    mean_rating = actual.mean()
    ax.axhline(mean_rating, color="#f59e0b", linestyle=":", linewidth=1.5, label=f"Mean rating ({mean_rating:.2f})")

    ax.set_xlabel("Actual Average Rating", fontsize=12)
    ax.set_ylabel("Predicted Average Rating", fontsize=12)
    ax.set_title("Actual vs Predicted Ratings", fontsize=14, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="upper left")

    textstr = f"MAE = {metrics['mae']:.3f}\nRMSE = {metrics['rmse']:.3f}\nR² = {metrics['r2']:.3f}"
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db", alpha=0.9))

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_rating_distribution(preds: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0.75, 5.5, 0.25)
    ax.hist(preds["actual"], bins=bins, alpha=0.7, color="#2563eb", edgecolor="white", label="Actual")
    ax.hist(preds["predicted"], bins=bins, alpha=0.5, color="#dc2626", edgecolor="white", label="Predicted")
    ax.set_xlabel("Average Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Actual vs Predicted Ratings", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    history = pd.read_csv(RESULTS_DIR / "training_history.csv")
    preds = pd.read_csv(RESULTS_DIR / "val_predictions.csv")
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    print("Generating plots...")
    plot_loss_curve(history, RESULTS_DIR / "loss_curve.png")
    plot_actual_vs_predicted(preds, metrics, RESULTS_DIR / "actual_vs_predicted.png")
    plot_rating_distribution(preds, RESULTS_DIR / "rating_distribution.png")
    print("Done.")


if __name__ == "__main__":
    main()
