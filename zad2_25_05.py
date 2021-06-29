import matplotlib.pyplot as plt
import pprint

order = list(reversed([(i+1)*20 for i in range(len(df_bf))]))
labels = []
for index, row in df_bf.iterrows():
    labels.append(str(row['LABEL']))


plt.figure(figsize=(40, 5))
plt.bar(labels, order, width=0.1)

plt.tight_layout()
plt.show()