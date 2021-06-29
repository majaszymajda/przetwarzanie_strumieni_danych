import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pprint


temp_data = np.loadtxt('./Test/X_test.txt')
columns_names = np.loadtxt('./features.txt', usecols=1, dtype='str')

df = pd.DataFrame(temp_data, columns=columns_names)
df.to_csv('frame.csv')
print(df)


def FindBestFeatures(data, n_components):
    X = data.copy()
    model = PCA(n_components=n_components).fit(X)
    X_pc = model.transform(X)

    n_pcs = model.components.shape[0]

    # best features indexes
    most_important = [np.abs(model_components[i]).argmax() for i in range(n_pcs)]

    # connect best features to labels
    most_important_names = [columns_names[most_important[i]] for i in range(n_pcs)]

    # make dict for pandas
    dic = {f'PC{i}': most_important_names[i] for i in range(n_pcs)}
    df = pd.DataFrame(dic.items(), columns=['PCA', 'LABEL'])

    return df, dic


df_bf, dic_bf = FindBestFeatures(df, 20)
print(df_bf)

order = list(reversed([(i+1)*20 for i in range(len(df_bf))]))
labels = []
for index, row in df_bf.iterrows():
    labels.append(str(row['LABEL']))

plt.figure(figsize=(40, 5))
plt.bar(labels, order, width=0.1)
plt.tight_layout()
plt.show()
