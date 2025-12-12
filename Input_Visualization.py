# Some sections of this code and documentation were generated with the assistance of ChatGPT (OpenAI, 2025).
# All code was reviewed, revised, and adapted by the project author.

import argparse
import os
from pathlib import Path

# Ensure matplotlib can write
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


def load_data(path):
    """Load dataset CSV file"""
    return pd.read_csv(path)


def save_fig(name):
    """Save figure into a 'figures' folder."""
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlations(df):
    """Correlation heatmap using numeric columns only."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("No numeric columns available for correlation heatmap.")
        return

    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    if HAS_SEABORN:
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
        plt.title("Correlation Heatmap (Numeric Features Only)")
    else:
        plt.imshow(corr.values, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Heatmap (Numeric Features Only)")
    save_fig("correlation_heatmap")


def plot_distributions(df, columns):
    """Plot distributions for selected numeric columns."""
    for col in columns:
        plt.figure(figsize=(8, 5))
        
        #bed and bath non continuous discrete counts
        if col in ["bed", "bath"]:
            if HAS_SEABORN:
                sns.histplot(df[col], binwidth=1)
            else:
                plt.hist(df[col].dropna(), bins=int(df[col].max() - df[col].min() + 1))
            title_adj = ""
        
        #continuous variable use bins=50
        else:
            if HAS_SEABORN:
                sns.histplot(df[col], bins=50)
            else:
                plt.hist(df[col].dropna(), bins=50)
            if col in ["price", "house_size"]:
                plt.xscale("log")
                plt.gca().xaxis.set_major_formatter(ScalarFormatter())
                title_adj = " (log scale)"
            else:
                title_adj = ""

        #add titles
        plt.title(f"Distribution of {col}{title_adj}")
        
        #label x & y axis (with units where appropriate)
        if col == "price":
            plt.xlabel("Price (USD)")
        elif col == "house_size":
            plt.xlabel("House Size (sq ft)")
        else:
            plt.xlabel(col)
        plt.ylabel("Count")
        
        #avoid titles cut off
        plt.tight_layout()
        
        #save figures 
        save_fig(f"distribution_{col}")

def preprocess_data(df):
    """Select and clean relevant columns for visualization/modeling.

    This project uses `data/usa_housing.csv` with columns like:
    price, bedrooms, bathrooms, sqft_living, city, statezip, ...
    """
    df_model = df.copy()

    # Standardize to these feature names (used below)
    if "bedrooms" in df_model.columns and "bed" not in df_model.columns:
        df_model["bed"] = df_model["bedrooms"]
    if "bathrooms" in df_model.columns and "bath" not in df_model.columns:
        df_model["bath"] = df_model["bathrooms"]
    if "sqft_living" in df_model.columns and "house_size" not in df_model.columns:
        df_model["house_size"] = df_model["sqft_living"]

    # Parse state + zip from "statezip" (ex: "WA 98103")
    if "statezip" in df_model.columns:
        parts = df_model["statezip"].astype(str).str.split(r"\s+", n=1, expand=True)
        if "state" not in df_model.columns:
            df_model["state"] = parts[0]
        if "zip_code" not in df_model.columns:
            df_model["zip_code"] = parts[1] if parts.shape[1] > 1 else None

    required = ["price", "city", "state", "zip_code", "bed", "bath", "house_size"]
    missing = [c for c in required if c not in df_model.columns]
    if missing:
        raise SystemExit(
            f"Dataset is missing required columns for this script: {missing}\n"
            f"Available columns: {df_model.columns.tolist()}"
        )

    df_model = df_model[required].copy()
    df_model = df_model.dropna(subset=required)

    # convert categorical columns to strings
    df_model["zip_code"] = df_model["zip_code"].astype(str)
    df_model["state"] = df_model["state"].astype(str)
    df_model["city"] = df_model["city"].astype(str)

    #Remove outliers
    df_model = df_model[df_model["bed"] < 20]
    df_model = df_model[df_model["bath"] < 20]
    df_model = df_model[df_model["house_size"] < 20000]
    df_model = df_model[df_model["price"] < 5_000_000]

    return df_model
    
def main():
    parser = argparse.ArgumentParser(description="Generate EDA plots for usa_housing.csv")
    parser.add_argument(
        "--data",
        type=str,
        default="data/realtor-data.zip.csv",
        help="Path to dataset CSV (default: data/realtor-data.zip.csv).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Max rows to load for EDA (default: 200000).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data, nrows=args.max_rows)
    df = preprocess_data(df)

    # correlation heatmap
    plot_correlations(df)

    # Numeric columns
    numeric_cols = df.select_dtypes(include="number").columns[:5]
    plot_distributions(df, numeric_cols)

    print("All visualizations saved to /figures folder.")


if __name__ == "__main__":
    main()