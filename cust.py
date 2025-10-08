# -----------------------------
# cust.py - Customer Segmentation using K-Means
# -----------------------------

# Step 0: Force single-threading to avoid macOS MKL/BLAS issues
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # You can switch to MiniBatchKMeans if needed
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv('Mall_Customers.csv')  # Make sure the file is in the same folder
print("First 5 rows of the dataset:")
print(df.head())

# -----------------------------
# 3. Select features for clustering
# -----------------------------
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# 4. Standardize features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. Apply K-Means clustering
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)  # n_init to avoid warnings
df['Cluster'] = kmeans.fit_predict(X_scaled)

# -----------------------------
# 6. View cluster assignments
# -----------------------------
print("\nCluster assignments:")
print(df[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])

# -----------------------------
# 7. Visualize clusters
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='viridis',
    s=100
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title='Cluster')
plt.show()
