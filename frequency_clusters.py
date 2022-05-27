import pandas as pd
import numpy as np
import json
import logging
from sklearn.cluster import AgglomerativeClustering


#@click.command()
#@click.option('-o', '--output', 'opath', default='', help='path to save plot figures', required=True)
#@click.option('-i', '--input', 'ipath', help='path to fetch mapping and embeds in json format', required=True)
#@click.option('-c', '--cluster-algorithm', 'c', default='slink', help='cluster algorithm to be used: tlsh / k-means / slink', show_default=True)
#@click.option('-cf', '--config-file', 'cf', help='json configuration file containing binaryName: malwareClass', required=True)

# configure logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

class Cluster:
    def __init__(self, classNamesList):
        self.__classDistribution = {}
        for className in classNamesList:
            self.__classDistribution[className] = 0.0
    def update(self, className: str) -> None:
        if className in self.__classDistribution:
            self.__classDistribution[className] += 1
    def extractClassName(self) -> str:
        return max(self.__classDistribution, key=self.__classDistribution.get)

def predict_classes(df, X, binaryToClass, linkage):
    clusters = []
    classes = set(binaryToClass.values())
    n_classes  = len(classes)
    logging.info(f'starting clustering with {n_classes} clusters ...')
    labels = AgglomerativeClustering(n_clusters=n_classes, affinity='cosine', linkage=linkage).fit(X).labels_
    for i in range(n_classes):
        cluster = Cluster(classes)
        indexes = np.where(labels==i)[0]
        for idx in indexes:
            binary = df.loc[idx, 'binary']
            if binary in binaryToClass:
                fn_class = binaryToClass[binary]
                cluster.update(fn_class)
        clusters.append(cluster)
    predicted_classes = [clusters[label].extractClassName() for i, label in enumerate(labels)] 
    return predicted_classes

def plot_binary_freq(df, binary, labels):
    data = df[df['binary'] == binary]
    g = sns.barplot(data=data, x='labels', y='freq', order=labels)
    g.set(title=binary, ylim=(0, 1.2))
    fig = g.get_figure()
    fig.savefig(f'test/{binary}.png')
    fig.clf()

def cli():
    logging.info('loading the configuration file ...')
    # binaryToClass = {'wireshark': 'zbot', ...}
    binaryToClass = json.load(open('binaryToClass.json', 'r'))
    
    logging.info('loading embedding and mapping file ...')
    loaded_data = json.load(open('test_vectors_integrati.json', 'r'))
    X = np.array(loaded_data['embeddings'])
    mapping_data = loaded_data['function_mapper']
    
    data = []
    for i in range(0, mapping_data['size']):
        data.append(list(mapping_data['map_embds_to_fn'][str(i)].values())) 
    feat_cols = ['feat'+str(i) for i in range(X.shape[1])]
    df = pd.concat([pd.DataFrame(X, columns=feat_cols), 
                    pd.DataFrame(data, columns=['name', 'binary'])], axis=1)
    df['labels'] = ''
    for binary in binaryToClass:
        df.loc[df['binary'] == binary, 'labels'] = binaryToClass[binary]
        #print(f"[+] {df_full.loc[df_full['labels'] == '', 'labels'].count()}")
    
    logging.info('predicting classes ...')
    classes = predict_classes(df, X, binaryToClass, 'single')
    for i, clss in enumerate(classes):
        if df.loc[i, 'labels'] == '':
            df.loc[i, 'labels'] = clss
    
    logging.info('starting TSNE ...')
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(df[feat_cols].values)
    df['tsne_1'] = X_tsne[:, 0]
    df['tsne_2'] = X_tsne[:, 1]
    fig = sns.scatterplot(
        x="tsne_1", y="tsne_2",
        data=df,
        hue=df['labels'],
        legend="full",
        alpha=0.3,
     ).get_figure()
    fig.savefig('clusters.png')
    fig.clf()
    logging.info(f'saved plot to clusters.png')
    
    df = df.loc[:, ('binary', 'labels')]
    df['freq_count'] = df.groupby(['binary', 'labels'])['binary'].transform('count')
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['n_count'] = df['binary'].map(df.groupby('binary')['freq_count'].sum())
    df['freq'] = df.apply(lambda x: x['freq_count']/x['n_count'],axis=1)
    df.sort_values(by=['binary', 'freq'], ascending=False)

    labels = df['labels'].unique()
    for binary in df['binary'].unique():
        logging.info(f'creating plot for binary {binary} ...')
        plot_binary_freq(df, binary, labels)
    

if __name__ == '__main__':
    cli()
