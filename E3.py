# 3. PCA to reduce dimensionality of iris dataset from 4 to 2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target

df_before = pd.DataFrame(X, columns = iris.feature_names)
df_before["Species"] = iris.target_names[y]

print("IRIS DATASET BEFORE PCA(Labeled): \n")
print(df_before.head(10))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

df_after = pd.DataFrame(X_pca, columns = ['Principle Component 1', 'Principle Component 2'])
df_after['Species'] = iris.target_names[y]

print("\n IRIS DATASET AFTER PCA(Labeled): \n")
print(df_after.head(10))

plt.figure(figsize=(8, 6))
for species in df_before['Species'].unique():
  subset = df_before[df_before['Species'] == species]
  plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label = species)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title("IRIS DATASET BEFORE PCA")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
for species in df_after['Species'].unique():
  subset = df_after[df_after['Species'] == species]
  plt.scatter(subset['Principle Component 1'], subset['Principle Component 2'], label = species)

plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.title("IRIS DATASET BEFORE PCA")
plt.legend()
plt.grid(True)
plt.show()

print("\nExplained variace Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", np.sum(pca.explained_variance_ratio_))
