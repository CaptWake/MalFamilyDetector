"""
    TO-DO:
        - Scrivere il codice per esportare in csv o json o npz 
        - aggiungere export di y
        - refactoring
"""
import pandas as pd
import numpy as np
import ember

import utils
import file_parser
import pathlib
import hashlib
import json

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
        X = PEDataset().load_raw_from_file(X_file)
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
    
    #def export_dataset(self, filename):   
    
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
    def get_X(self):
        return np.vstack(self.dataset['features'])
    
    def export_dataset(self, filename):
        exported = self.dataset.to_json(orient="records")
        with open(filename, 'w+') as f:
            json.dump(json.loads(exported), f)
    
    def data_cleansing(self):
        self.dataset = pd.DataFrame(self.dataset, columns = ['sha256', 'features']).drop_duplicates(subset=['sha256'])
        return self
    
    def representation_transformation(self):
        features = np.vstack(self.dataset['features'])
        self.dataset['features'] = self._normalize(features).tolist()
        return self

    
class TrainTestPEPreprocessing(PEPreprocessing):
    
    def get_X_y(self):
        return np.vstack(self.dataset['X']['features']), self.dataset['y']
    
    def export_dataset(self, filename):
        self.dataset['X'] = self.dataset['X'].to_json(orient="records")
        with open(filename, 'w+') as f:
            json.dump(json.loads(self.dataset['X']), f)
    
    def data_cleansing(self):
        # removes duplicates
        self.dataset['X'] = pd.DataFrame(self.dataset['X'], columns = ['sha256', 'features']).drop_duplicates(subset=['sha256'])
        self.dataset['y'] = self.dataset['y'][(self.dataset['y']['sha'].isin(self.dataset['X']['sha256']))]
        # added now
        return self
    
    def representation_transformation(self):
        features = np.vstack(self.dataset['X']['features'])
        self.dataset['X']['features'] = self._normalize(features).tolist()
        self.dataset['y']['family'] = self.__prepare_targets(list(self.dataset['y']['family']))
        print(self.dataset['y'])
        return self

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
        #print(inv_mapping)
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
        X = pd.DataFrame(X) 
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
        return features

    def __extract_dataset_from_file(self, file):
        df = None 
        ext = pathlib.Path(file).suffix
        if(ext == '.json'):
            df = file_parser.JSONcreator().create_parser().parse(file)
        elif(ext == '.csv'):
            df = file_parser.CSVcreator().create_parser().parse(file)
        #elif(ext == '.npz'):
        #    df = pd.DataFrame(np.load(file), columns=['X']
        return df
    
    
def main():
    #PEPreprocessingBuilder().from_existing_train_test_dataset('/home/students/derosa/test/', '/home/students/derosa/bodmas/bodmas_metadata.csv', '/home/students/derosa/config.json')
    pe1 = PEPreprocessingBuilder().new_train_test_dataset('/home/students/derosa/test/', '/home/students/derosa/bodmas/bodmas_metadata.csv', '/home/students/derosa/config.json')
    pe2 = PEPreprocessingBuilder().from_existing_raw_dataset('/home/students/derosa/bodmas/BODMAS/code/bodmas/prova.json')
    X = pe2.data_cleansing().representation_transformation().get_X()
    #X = pe2.data_cleansing().get_X()
    lgbm_model = lgb.Booster(model_file=MODEL_FOLDER + 'gbdt_bluehex_families_238_r10.txt')
    print(predict(lgbm_model, 'gbdt', X, DATA_FOLDER + 'top_238_label_mapping.json'))
    #print(type(pe1.data_cleansing().representation_transformation().get_dataset()))
    #print(pe1.get_X_y())
    #pe2.export_dataset('prova2.json')
    pe1.data_cleansing().representation_transformation().export_dataset('prova4.json')
    
    
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
