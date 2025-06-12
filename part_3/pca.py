# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import pandas as pd
# from preprocessing.preprocess_data import preprocess
#
#
# df = pd.read_csv('../train_test_splits/train_split.feats.csv',
#                      encoding='utf-8-sig')
# df = preprocess(df, normalize=True)
#
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(df)  # scaled_data: after normalization
#
# plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA - First Two Components')
# plt.show()
#
#
# kmeans = KMeans(n_clusters=5, random_state=42)
# clusters = kmeans.fit_predict(df)
# df['cluster'] = clusters
#
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6)
# plt.title('Clusters visualized in PCA space')
# plt.show()
#
# print("Explained variance ratio:", pca.explained_variance_ratio_)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting
import numpy as np
from preprocessing.preprocess_data import preprocess
import pandas as pd

df = pd.read_csv('../train_test_splits/train_split.feats.csv',
                     encoding='utf-8-sig')
df = preprocess(df, normalize=True)

tumor_size = pd.read_csv('../train_test_splits/train_split.labels.1.csv', encoding='utf-8-sig')
metastasis = pd.read_csv("../train_test_splits/train_split.labels.0.csv", encoding='utf-8-sig')

# metastasis[0] =

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(df)

# Perform PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df)

# Normalize tumor size to use as dot size
# Avoid extreme sizes by clipping or using log scale if needed
tumor_size_normalized = (tumor_size - tumor_size.min()) / (tumor_size.max() - tumor_size.min())
dot_sizes = 100 * tumor_size_normalized + 10  # scale for visibility

# Randomly select 20% of the data
np.random.seed(42)  # for reproducibility
sample_indices = np.random.choice(len(X_pca), size=int(0.5 * len(X_pca)), replace=False)

# Subset the data
X_pca_sampled = X_pca[sample_indices]
dot_sizes_sampled = dot_sizes.iloc[sample_indices]
clusters_sampled = clusters[sample_indices]

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    X_pca_sampled[:, 0],
    X_pca_sampled[:, 1],
    X_pca_sampled[:, 2],
    s=dot_sizes_sampled,
    c=clusters_sampled,
    cmap='tab10',
    alpha=0.7
)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA: Tumor Size (dot size), Cluster (color)')
plt.colorbar(sc, ax=ax, label='Cluster')
plt.tight_layout()
plt.show()
