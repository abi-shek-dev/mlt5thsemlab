import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv('7th/your_data.csv')

data = df.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
X = scaler.fit_transform(data)

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm_labels = gmm.fit_predict(X)

sil_kmeans = silhouette_score(X, kmeans_labels)
sil_gmm = silhouette_score(X, gmm_labels)

print("\nClustering Evaluation:")
print(f"K-Means Silhouette Score: {sil_kmeans:.4f}")
print(f"EM (GMM) Silhouette Score: {sil_gmm:.4f}")

X_pca = PCA(n_components=2).fit_transform(X)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="Set2", ax=axs[0])
axs[0].set_title("K-Means Clustering")

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=gmm_labels, palette="Set1", ax=axs[1])
axs[1].set_title("EM (GMM) Clustering")

plt.tight_layout()
plt.show()