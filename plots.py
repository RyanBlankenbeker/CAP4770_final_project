import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    
     # Plot 1 — Distribution of Delays
    plt.figure(figsize=(7, 5))
    sns.countplot(x="delayed", data=df)
    plt.title("Flight Delay Distribution")
    plt.xlabel("Delayed (1=yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plot_delay_distribution.png")

    # Plot 2 — Delays by Airport
    plt.figure(figsize=(10, 5))
    sns.barplot(x="airport", y="arr_delay", data=df)
    plt.title("Average Arrival Delay by Airport")
    plt.tight_layout()
    plt.savefig("plot_delay_by_airport.png")

    # Plot 3 — Heatmap of correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plot_correlation_heatmap.png")

    print("EDA plots saved.")
