from sklearn.decomposition import PCA

def findBestFeatures(data, n_components):
    X = data.copy()
    model = PCA(n_components=n_components).fit(X)
    X_pc = model.transform(X)

    n_pcs = model.components_.shape[0]

    # best features indexes
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    # connect best features to labels
    most_important_names = [columns_names[most_important[i]] for i in range(n_pcs)]

    # make dict for pandas
    dic = {f'PC{i}': most_important_names[i] for i in range(n_pcs)}
    df = pd.DataFrame(dic.items(), columns=['PCA', 'LABEL'])

    return df, dic

df_bf, dic_bf = findBestFeatures(df, 20)
print(df_bf)
