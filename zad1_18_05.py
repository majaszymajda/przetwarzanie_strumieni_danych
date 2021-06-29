import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

temp_data = np.loadtxt('./Test/X_test.txt')
columns_names = np.loadtxt('./features.txt', usecols=1, dtype='str')

df = pd.DataFrame(temp_data, columns=columns_names)
df.to_csv('frame.csv')


X_test = df.copy()
pca = PCA(0.95)
X_pca = pca.fit_transform(X_test)
print(pca.n_components_)

