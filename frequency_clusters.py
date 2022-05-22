import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = None
df2 = None

with open('vec.json') as f:
    X = json.load(f) 
with open('mapping.json') as f:
    t = json.load(f)
    tt = []
    for i in range(0, t['size']):
        tt.append(list(t['map_embds_to_fn'][str(i)].values())) 
    df2 = pd.DataFrame(tt, columns=['name', 'binary'])
X = np.array(X)
feat_cols = [ 'feat'+str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)

df_full = pd.concat([df, df2], axis=1)
clustering = AgglomerativeClustering(n_clusters=4).fit(X)
df_full['labels'] = clustering.labels_

tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(df[feat_cols].values)
df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    data=df,
    hue=clustering.labels_,
    legend="full",
    alpha=0.3
)

df1 = df_full.loc[:, ('binary', 'labels')]
df1['freq_count'] = df1.groupby(['binary', 'labels'])['binary'].transform('count')
df1.drop_duplicates(keep='first', inplace=True)
df1.reset_index(drop=True, inplace=True)
df1['n_count'] = df1['binary'].map(df1.groupby('binary')['freq_count'].sum())
df1['freq'] = df1.apply(lambda x: x['freq_count']/x['n_count'],axis=1)
df1
