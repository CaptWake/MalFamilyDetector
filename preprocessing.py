"""
    TO-DO:
        - Aggiungere la possibilità di salvare il dataset generato in npz o csv -> metodo di PEPreprocessing
        - Aggiungere la possibilità di fare features selection
"""
import pandas as pd
import numpy as np
import ember

import utils
import file_parser
import pathlib
import hashlib

from collections import OrderedDict 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# togliere poi, usata solo per predict
DATA_FOLDER = '/home/students/derosa/bodmas/BODMAS/code/multiple_data/bluehex_multiclass/'
MODEL_FOLDER = '/home/students/derosa/bodmas/BODMAS/code/multiple_models/bluehex_multiclass/'
import lightgbm as lgb

class PEPreprocessingBuilder: 
    def from_existing_train_test_dataset(self, X_file, y_file):
        X, y = PEDataset().load_test_train_from_file(X_file, y_file)
        tpe = TrainTestPEPreprocessing()
        tpe.set_dataset({'X': X, 'y': y})
        return tpe
    
    def new_train_test_dataset(self, dir_name,  y_file, features_file):
        X, y = PEDataset().generate_test_train(dir_name, y_file, features_file)
        tpe = TrainTestPEPreprocessing()
        tpe.set_dataset({'X': X, 'y': y})
        return tpe
    
    def from_existing_raw_dataset(self, X_file):
        X = PEDataset().load_raw_from_file(X_file, y_file)
        rpe = RawPEPreprocessing()
        rpe.set_dataset(X)
        return rpe
    
    def new_raw_dataset(self, dir_name, features_file):
        X = PEDataset().generate_raw(dir_name, features_file)
        rpe = RawPEPreprocessing()
        rpe.set_dataset(X)
        return rpe


class PEPreprocessing:
    
    dataset = None
    
    def set_dataset(self, X):
        self.dataset = X
    
    def get_dataset(self):
        return self.dataset
    
    #@abstractmethod
    def data_cleansing(self):
        pass
        
    #@abstractmethod
    def feature_tuning(self):
        pass
    
    #@abstractmethod
    def representation_transformation(self):
        pass

    ''' 
    Scale features to a standard range so that all values are within the new range of 0 and 1. 
    Useful for models that use a weighted sum of input variables i.e SVM, MLP, KNN
    '''
    def _normalize(self, X):
        scaler = MinMaxScaler()
        X_scale = scaler.fit_transform(X)
        return X_scale
    
        
class RawPEPreprocessing(PEPreprocessing):
    def data_cleansing(self):
        self.dataset = pd.DataFrame(self.dataset, columns = ['sha256', 'features']).drop_duplicates(subset=['sha256'])
        self.dataset = np.vstack(self.dataset['features'])
        return self.dataset
    
    def representation_transformation(self):
        self.dataset = self._normalize(self.dataset)
        return self.dataset

    
class TrainTestPEPreprocessing(PEPreprocessing):

    def data_cleansing(self):
        # removes duplicates
        hash_list = list(OrderedDict.fromkeys([row['sha256'] for row in self.dataset['X']]))
        self.dataset['X'] = pd.DataFrame(self.dataset['X']).drop_duplicates(subset=['sha256'])
        # non so se ci vada o meno testiamo tra poco
        #self.dataset['X'] = self.dataset['X'][(self.dataset['X']['sha256'].isin(hash_list))]
        self.dataset['y'] = self.dataset['y'][(self.dataset['y']['sha'].isin(hash_list))]
        self.dataset['X'] = self.dataset['X'].drop(columns=['sha256'])
        # transform dataframe to numpy
        # -----
        self.dataset['X'] = np.vstack(self.dataset['X']['features'].to_numpy())
        # ---
        self.dataset['y'] = list((self.dataset['y'].drop(columns=['sha', 'timestamp']))['family'])
        return self.dataset['X'], self.dataset['y']
        
    def representation_transformation(self):
        self.dataset['X'] = self._normalize(self.dataset['X'])
        self.dataset['y'] = self.__prepare_targets(self.dataset['y'])
        return self.dataset['X'], self.dataset['y']

    ''' 
    Transform training set to continuous labels using ordinal encoding cause the 
    families names are uncorrelated.
    '''
    def __prepare_targets(self, y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.__create_mapping_file(y, y_enc)
        return y_enc

    def __create_mapping_file(self, raw, enc):
        mapping = {}
        inv_mapping = {}
        for i in range(len(raw)):
            mapping[raw[i]] = enc[i]  # mapping: real label -> converted label
            inv_mapping[str(enc[i])] = raw[i] # inv_mapping: converted label -> real label
        print(inv_mapping)
        #logging.debug(f'LabelEncoder mapping: {mapping}')
        #logging.debug(f'after relabeling training: {Counter(y_train)}')
        #utils.dump_json(inv_mapping, setting_data_folder, f'top_{families_cnt}_label_mapping.json')
    
    
class PEDataset:
    def generate_raw(self, dir_name, features_file):
        X = []
        files = pathlib.Path(dir_name).glob('*')
        print(files)
        for file in files:
            X.append({'sha256': calculate_sha256(file), 'features': self.__extract_features_from_binary(file, features_file)})
        return X
    
    def load_raw_from_file(self, X_file):
        return self.__extract_dataset_from_file(X_file)
    
    def generate_test_train(self, dir_name, y_file, features_file):
        X = []
        files = pathlib.Path(dir_name).glob('*')
        for file in files:
            X.append({'sha256': calculate_sha256(file), 'features': self.__extract_features_from_binary(file, features_file)})
        y = self.__extract_dataset_from_file(y_file)
        return X, y
    
    def load_test_train_from_file(self, X_file, y_file):
        X = self.__parse_dataset_from_file(X_file)
        y = self.__parse_dataset_from_file(y_file)
        return X, y
    
    def __extract_features_from_binary(self, binary_name, features_file=''):
        file_data = open(binary_name, 'rb').read()
        extractor = ember.features.PEFeatureExtractor(2, features_file=features_file)
        features = np.array(extractor.feature_vector(file_data))
        #X = features[:,np.newaxis]
        #print(X.T.shape)
        #return X.T
        return features

    def __extract_dataset_from_file(self, file):
        df = None 
        ext = pathlib.Path(file).suffix
        if(ext == '.json'):
            df = file_parser.JSONcreator().create_parser().parse(file)
        elif(ext == '.csv'):
            df = file_parser.CSVcreator().create_parser().parse(file)
        return df
    
    
def main():
    #pe = PEPreprocessingBuilder().new_train_test_dataset('/home/students/derosa/test/', '/home/students/derosa/bodmas/bodmas_metadata.csv', '/home/students/derosa/config.json')
    pe = PEPreprocessingBuilder().new_raw_dataset('/home/students/derosa/test/', '/home/students/derosa/config.json')
    print(pe.data_cleansing())
    X = pe.representation_transformation()
    lgbm_model = lgb.Booster(model_file=MODEL_FOLDER + 'gbdt_bluehex_families_238_r10.txt')
    print(predict(lgbm_model, 'gbdt', X, DATA_FOLDER + 'top_238_label_mapping.json'))



'''
def normalize(X_train, X_test_list):
    scaler = MinMaxScaler()

    X_train_scale = scaler.fit_transform(X_train)
    logging.debug(f'X_train_scale: {X_train_scale.shape}')

    X_test_scale_list = []
    logging.debug(f'X_test_list: {len(X_test_list)}, X_test_list[0]: {X_test_list[0].shape}')
    for X_test in X_test_list:
        X_test_scale = scaler.transform(X_test)
        X_test_scale_list.append(X_test_scale)

    return X_train_scale, X_test_scale_list
'''

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
    return [mapping[f'{prediction.item(0)}'] for prediction in y_pred]

def calculate_sha256(filename):
    sha256_hash = hashlib.sha256()
    with open(filename,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
        return(sha256_hash.hexdigest())

if __name__ == '__main__':
    main()
