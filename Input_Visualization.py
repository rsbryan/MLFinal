# Some sections of this code and documentation were generated with the assistance of ChatGPT (OpenAI, 2025).
# All code was reviewed, revised, and adapted by the project author.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


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
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features Only)")
    save_fig("correlation_heatmap")


def plot_distributions(df, columns):
    """Plot distributions for selected numeric columns."""
    for col in columns:
        plt.figure(figsize=(8, 5))
        
        #bed and bath non continuous discrete counts
        if col in ["bed", "bath"]:
            sns.histplot(df[col], binwidth=1)
            title_adj = ""
        
        #continuous variable use bins=50
        else:
            sns.histplot(df[col], bins=50)
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
    """Select and clean relevant columns."""
    selected_cols = ["price", "state", "city", "zip_code", "bed", "bath", "house_size"]
    
    df_model = df[selected_cols].copy()
    df_model = df_model.dropna(subset=selected_cols)

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
    
    data_path = "realtor-data.zip.csv" 

    df = load_data(data_path)
    df = preprocess_data(df)

    # correlation heatmap
    plot_correlations(df)

    # Numeric columns
    numeric_cols = df.select_dtypes(include="number").columns[:5]
    plot_distributions(df, numeric_cols)

    print("All visualizations saved to /figures folder.")


if __name__ == "__main__":
    main()