from sklearn.decomposition import PCA

X_test = df.copy()
pca = PCA(0.95)
X_pca = pca.fit_transform(X_test)
print(pca.n_components_)
