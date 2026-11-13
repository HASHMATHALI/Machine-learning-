"""Customer segmentation with K-Means on the Mall Customers dataset.

    python cust.py            # choose k by silhouette score, write figures and a profile table
    python cust.py --k 5      # force a number of clusters

Outputs (next to this script): elbow_silhouette.png, clusters.png,
cluster_profiles.csv and segments.csv.
"""
import argparse
import os
import sys

# Single-threaded BLAS avoids a KMeans memory-leak warning on Windows and macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")  # save figures instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def scan_k(X_scaled, k_values, seed=42):
    """Inertia (for the elbow plot) and silhouette score for each k."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(X_scaled, model.labels_)})
    return pd.DataFrame(rows)


def segment(df, k, seed=42):
    """Fit K-Means with k clusters; returns labelled rows and a per-cluster profile."""
    X_scaled = StandardScaler().fit_transform(df[FEATURES])
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_scaled)
    out = df.copy()
    out["Cluster"] = labels
    profile = out.groupby("Cluster")[FEATURES].mean().round(1)
    profile.insert(0, "Customers", out["Cluster"].value_counts().sort_index())
    return out, profile


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(HERE, "Mall_Customers.csv"))
    parser.add_argument("--k", type=int, default=None, help="number of clusters (default: best silhouette for k = 2..10)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X_scaled = StandardScaler().fit_transform(df[FEATURES])

    scan = scan_k(X_scaled, range(2, 11))
    best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    k = args.k or best_k
    print(scan.round(3).to_string(index=False))
    print(f"\nBest silhouette at k = {best_k}; using k = {k}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(scan["k"], scan["inertia"], marker="o"); ax1.set_title("Elbow: inertia by k"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax2.plot(scan["k"], scan["silhouette"], marker="o"); ax2.axvline(k, linestyle="--", color="grey")
    ax2.set_title("Silhouette score by k"); ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "elbow_silhouette.png"), dpi=120); plt.close(fig)

    labelled, profile = segment(df, k)
    print("\nCluster profiles (means):")
    print(profile.to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(labelled["Annual Income (k$)"], labelled["Spending Score (1-100)"], c=labelled["Cluster"], cmap="viridis", s=60)
    ax.set_title(f"Customer segments (K-Means, k = {k})"); ax.set_xlabel("Annual income (k$)"); ax.set_ylabel("Spending score (1-100)")
    ax.legend(*points.legend_elements(), title="Cluster")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "clusters.png"), dpi=120); plt.close(fig)

    profile.to_csv(os.path.join(HERE, "cluster_profiles.csv"))
    labelled.to_csv(os.path.join(HERE, "segments.csv"), index=False)
    print("\nWrote elbow_silhouette.png, clusters.png, cluster_profiles.csv and segments.csv")


if __name__ == "__main__":
    main()
