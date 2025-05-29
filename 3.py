import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris=load_iris()
pca=PCA(n_components=2)
pca_result=pca.fit_transform(iris.data)

plt.figure(figsize=(8,6))
colors=['r','g','b']
target_names=iris.target_names

for i, color in enumerate(colors):
    plt.scatter(
        pca_result[iris.target==i,0],
        pca_result[iris.target==i,1],
        color=color,
        label=target_names[i]
    )

plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()