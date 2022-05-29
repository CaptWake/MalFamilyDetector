import pandas as pd
import numpy as np
import seaborn as sns
import json
import logging
import os
from pathlib import Path
import click
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

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

def predict_classes_KMeans(params, df, X, binaryToClass):
    clusters = []
    classes = set(binaryToClass.values())
    n_classes  = len(classes)
    logging.info(f'starting KMeans clustering ...')
    labels = KMeans(**params).fit(X).labels_
    n_classes = len(set(labels))
    for i in range(0, n_classes):
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

def predict_classes_DBSCAN(params, df, X, binaryToClass):
    clusters = []
    classes = set(binaryToClass.values())
    logging.info(f'starting DBSCAN clustering ...')
    labels = DBSCAN(**params).fit(X).labels_
    n_classes = len(set(labels)) - (1 if -1 in labels else 0)
    logging.info(f'estimated number of clusters: {n_classes}')
    for i in range(0, n_classes):
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

def predict_classes_naive(df, X, binaryToClass):
    clusters = []
    classes = set(binaryToClass.values())
    n_classes  = len(classes)
    logging.info(f'starting clustering with {n_classes} clusters ...')
    labels = AgglomerativeClustering(n_clusters=n_classes, affinity='cosine', linkage='single').fit(X).labels_
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

def plot_binary_freq(df, binary, labels, opath):
    data = df[df['binary'] == binary]
    g = sns.barplot(data=data, x='labels', y='freq', order=labels)
    g.set(title=binary, ylim=(0, 1.2))
    fig = g.get_figure()
    fig.savefig(opath / f'{binary}.png')
    fig.clf()

@click.command()
@click.option('-legacy', '--legacy-clustering', 'l', default=False, help='specify if wanna use the naive approach to identify clusters')
@click.option('-cf', '--config-file', 'cf', help='json configuration file', required=True)

def cli(legacy, cf):

    
    # configure logging 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    logging.info('loading the configuration file ...')
    cfg = json.load(open(cf, 'r'))
    opath = Path(cfg['output_path'])

    if not os.path.exists(opath):
        os.mkdir(opath)

    # binaryToClass = {'wireshark': 'zbot', ...}
    binaryToClass = json.load(open(cfg['binary2class'], 'r'))
    
    logging.info('loading embedding and mapping file ...')
    loaded_data = json.load(open(cfg['dataset'], 'r'))
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
    
    logging.info('predicting classes ...')
    if legacy:
        classes = predict_classes_naive(df, X, binaryToClass)
    else:
        model_info = cfg['model']
        clustering_model_name = model_info['name']
        clustering_functions = { 
                            'DBSCAN': predict_classes_DBSCAN,
                            'KMeans': predict_classes_KMeans
                            }
        clustering_function = clustering_functions[clustering_model_name]
        classes = clustering_function(model_info['params'], df, X, binaryToClass)
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
    fig.savefig(opath / 'clusters.png')
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
        plot_binary_freq(df, binary, labels, opath)
    

if __name__ == '__main__':
    cli()
