import os
import argparse

import numpy as np
import ember

def main():
    parser = argparse.ArgumentParser(description='Ember Feature Extraction')
    parser.add_argument('binary', type=str,
                        help='PE file to extract features from')
    parser.add_argument('-o', dest='outfile', type=str, default='/tmp/features.npz',
                        help='output npz stored features file')
    args = parser.parse_args()
    if args.binary:
        features = extract_features_from_binary(args.binary)
        np.savez_compressed(args.outfile, X=features)
        
def extract_features_from_binary(binary_name, features_file=''):
    file_data = open(binary_name, 'rb').read()
    extractor = ember.features.PEFeatureExtractor(2, features_file=features_file)
    features = np.array(extractor.feature_vector(file_data))
    X = features[:,np.newaxis]
    print(X.T.shape)
    return X.T


def test(model, clf, X_test, y_test):
    

def train(model, clf, X_train, y_train):    
    model.fit(X_train, y_train)

# mettere un parametro di mapping labels
def predict(model, clf, X, label_mapping_file):
    if clf == 'rf':
        y_pred = model.predict_proba(X)
    else:
        y_pred = model.predict(X)

    ''' use -1 to sort in descending order.
        another solution is to use np.flip(y_pred, axis=1)
        y_pred[::-1] would reverse with axis=0
    '''
    y_pred = np.argsort(-1 * y_pred, axis=1)[:, :1]
    
    print(f'y_pred shape: {y_pred.shape} = {y_pred}')
    mapping = utils.load_json(label_mapping_file)
    return mapping[f'{y_pred.item(0)}']

if __name__ == 'main':
    main()
