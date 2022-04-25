"""
    TO-DO:
        - Scrivere il codice per esportare e importare in npz 
        - Aggiungere messaggi di logging alle varie azioni di preprocessing 
        - Refactoring
        
        risolvere problema del csv che quando viene esportato salva il tutto come stringhe
        e questo implica che le liste devono essere convertite manualmente, altrimenti utilizzare pickle
        come suggerisce questo post di SO : https://stackoverflow.com/questions/49580996/why-do-my-lists-become-strings-after-saving-to-csv-and-re-opening-python
        qui c'è il codice di conversione di stringa -> lista manuale:
"""
import pandas as pd
import numpy as np
import ember

import utils
import file_parser
import pathlib
import hashlib
import json
import logging
from logger import init_log

from collections import OrderedDict 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# togliere poi, usata solo per predict
DATA_FOLDER = '/home/students/derosa/bodmas/BODMAS/code/multiple_data/bluehex_multiclass/'
MODEL_FOLDER = '/home/students/derosa/bodmas/BODMAS/code/multiple_models/bluehex_multiclass/'
import lightgbm as lgb

class PEPreprocessingBuilder:
    """
    This class allows the user to create the correct preprocessing interface 
    based on his particular needs.
    """
    def from_existing_train_test_dataset(self, X_filename, y_filename):
        """Creates a preprocessing interface containing the existing features and targets datasets

        Parameters
        ----------
        X_filename : str
            The file location of the features dataset
        y_filename : str
            The file location of the targets dataset

        Returns
        -------
        TrainTestPEPreprocessing
            a preprocessing interface used to generate the pre-processed datasets from the loaded datasets
        """
        X, y = PEDataset().load_test_train_from_file(X_filename, X_filename)
        logging.info(f'Loaded successfully features dataset {X_filename} and targets dataset {y_filename} ...')
        tpe = TrainTestPEPreprocessing()
        tpe.set_dataset({'X': X, 'y': y})
        return tpe
    
    def new_train_test_dataset(self, dir_name,  y_filename, features_filename):
        """Creates a preprocessing interface containing the existing targets dataset and the newly generated features dataset 

        Parameters
        ----------
        dir_name : str
            The directory location of the samples used for building the features dataset
        y_filename : str
            The file location of the targets dataset
        features_filename : str
            The file location of the configuration file containing the features to be extracted from the samples
        
        Returns
        -------
        TrainTestPEPreprocessing
            a preprocessing interface used to generate the pre-processed datasets from the loaded datasets
        """
        X, y = PEDataset().generate_test_train(dir_name, y_filename, features_filename)
        logging.info('Generated successfully features dataset ...')
        logging.info(f'Loaded successfully targets dataset {y_filename} ...')
        tpe = TrainTestPEPreprocessing()
        tpe.set_dataset({'X': X, 'y': y})
        return tpe
    
    def from_existing_raw_dataset(self, X_filename):
        """Creates a preprocessing interface containing the existing features dataset

        Parameters
        ----------
        X_filename : str
            The file location of the features dataset
        
        Returns
        -------
        RawPEPreprocessing
            a preprocessing interface used to generate the pre-processed dataset from the loaded dataset
        """
        X = PEDataset().load_raw_from_file(X_filename)
        logging.info(f'Loaded successfully features dataset {X_filename}...')
        rpe = RawPEPreprocessing()
        rpe.set_dataset(X)
        return rpe
    
    def new_raw_dataset(self, dir_name, features_filename):
        """Creates a preprocessing interface containing the newly generated features dataset

        Parameters
        ----------
        dir_name : str
            The directory location of the samples used for building the features dataset
        features_filename : str
            The file location of the configuration file containing the features to be extracted from the samples
        
        Returns
        -------
        RawPEPreprocessing
            a preprocessing interface used to generate the pre-processed dataset from the loaded dataset
        """
        X = PEDataset().generate_raw(dir_name, features_filename)
        logging.info('Generated successfully features dataset ...')
        rpe = RawPEPreprocessing()
        rpe.set_dataset(X)
        return rpe


class PEPreprocessing:
    """
    A generic preprocessing interface used to define which operations should have
    every specific preprocessing interface.
    """
    dataset = None
    
    def set_dataset(self, X):
        self.dataset = X
    
    def get_dataset(self):
        return self.dataset
    
    """
    Removing or correcting records with corrupted or invalid values from raw data, 
    as well as removing records that are missing a large number of columns.
    """
    def data_cleansing(self):
        pass
    
    """
    Improving the quality of a feature for ML, which includes scaling and normalizing numeric values,
    imputing missing values, clipping outliers, and adjusting values with skewed distributions.
    """
    def feature_tuning(self):
        pass
    
    """
    Transformations of training data can reduce the skewness of data as well as 
    the prominence of outliers in the data
    """
    def representation_transformation(self):
        pass
    
    """ 
    Scales features to a standard range so that all values are within the new range of 0 and 1. 
    Useful for models that use a weighted sum of input variables i.e SVM, MLP, KNN
    """
    def _normalize(self, X):
        scaler = MinMaxScaler()
        X_scale = scaler.fit_transform(X)
        return X_scale
    
        
class RawPEPreprocessing(PEPreprocessing):
    """
    A preprocessing interface focussed on manipulating the features dataset in order
    to obtain the pre-preprocessed dataset that'll be used to feed the classificator 
    for prediction purposes
    """
    def get_X(self):
        return np.vstack(self.dataset['features'])
    
    def export_dataset(self, X_filename, protocol='csv'):
        if protocol == 'csv':
            self.dataset.to_csv(X_filename, index=False)
        exported = self.dataset.to_json(orient="records")
        with open(X_filename, 'w+') as f:
            json.dump(json.loads(exported), f)
    
    def data_cleansing(self):
        self.dataset.dropna(axis=0)
        # remove missing values inside features vector
        for idx, vec in enumerate(self.dataset['features']):
            if pd.DataFrame(vec).isnull().values.any():
                self.dataset.drop(idx, inplace=True)
        logging.debug('Successfully removed missing values ...')
        self.dataset.drop_duplicates(subset=['sha256'], inplace=True)
        logging.debug('Successfully removed duplicates  ...')
        return self

    def representation_transformation(self):
        features = np.vstack(self.dataset['features'])
        self.dataset['features'] = self._normalize(features).tolist()
        logging.debug('Successfully normalized features vector values ...')
        return self

    
class TrainTestPEPreprocessing(PEPreprocessing):
    """
    A preprocessing interface focussed on manipulating the features and targets datasets in order
    to obatin the pre-processed datasets that'll be used to feed the classificator 
    for training and testing purposes
    """
    def get_X_y(self):
        return np.vstack(self.dataset['X']['features']), self.dataset['y']
    
    def export_dataset(self, X_filename, y_filename, protocol='csv'):
        if protocol == 'csv':
            self.dataset['X'].to_csv(X_filename, index=False)
            self.dataset['y'].to_csv(y_filename, index=False)
        else:
            X_json = self.dataset['X'].to_json(orient="records")
            y_json = self.dataset['y'].to_json(orient="records")
            with open(X_filename, 'w+') as f:
                json.dump(json.loads(X_json), f)
            with open(y_filename, 'w+') as f:
                json.dump(json.loads(y_json), f)
    
    def data_cleansing(self):
        self.dataset['X'].dropna(axis=0)
        # remove missing values inside features vector
        for idx, vec in enumerate(self.dataset['X']['features']):
            if pd.DataFrame(vec).isnull().values.any():
                self.dataset['X'].drop(idx, inplace=True) 
        logging.debug('Successfully removed missing values ...')
        self.dataset['X'].drop_duplicates(subset=['sha256'], inplace=True)
        self.dataset['y'].drop_duplicates(subset=['sha'], inplace=True)
        self.dataset['y'] = self.dataset['y'][(self.dataset['y']['sha'].isin(self.dataset['X']['sha256']))]
        logging.debug('Successfully removed duplicates  ...')
        return self
    
    def representation_transformation(self):
        features = np.vstack(self.dataset['X']['features'])
        self.dataset['X']['features'] = self._normalize(features).tolist()
        logging.debug('Successfully normalized features vector values ...')
        self.dataset['y']['family'] = self.__prepare_targets(list(self.dataset['y']['family']))
        logging.debug('Successfully encoded elabels ...')
        return self

    """ 
    Transform training set to continuous labels using ordinal encoding cause the 
    families names are uncorrelated.
    """
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
        logging.debug(f'LabelEncoder mapping: {mapping}')
        with open('label_mapping.json', 'w+') as f:
            json.dump(inv_mapping, f)
    
    
class PEDataset:
    def generate_raw(self, dir_name, features_filename):
        """Creates a features vector dataframe from raw binaries inside a folder

        Parameters
        ----------
        dir_name : str
            The folder location containing the binaries
        features_filename: str
            The file location used for feature selection
            
        Returns
        -------
        DataFrame
            a dataframe containing the following columns:
                sha256 | features
        """
        X = []
        files = pathlib.Path(dir_name).glob('*')
        print(files)
        for file in files:
            X.append({'sha256': calculate_sha256(file), 'features': self.__extract_features_from_binary(file, features_filename)})
        return pd.DataFrame(X)
    
    def load_raw_from_file(self, X_filename):
        """Loads an features vector dataframe

        Parameters
        ----------
        X_filename : str
            The file location of the features dataset, it must be in the form of csv or json
            
        Returns
        -------
        DataFrame
            a dataframe containing the loaded features dataset
        """
        return self.__extract_dataset_from_file(X_filename)
    
    def generate_test_train(self, dir_name, y_filename, features_filename):
        """"Creates a features vector dataframe from raw binaries inside a folder

        Parameters
        ----------
        dir_name : str
            The folder location containing the binaries
       y_filename : str
            The file location of the targets dataset, it must be in the form of csv or json
        features_filename: str
            The file location used for feature selection
        
        Returns
        -------
        list<DataFrame>
            a list containing a dataframe that represents the extracted features vectors from binaries 
            and a dataframe representing targets variable
        """
        X = []
        files = pathlib.Path(dir_name).glob('*')
        for file in files:
            X.append({'sha256': calculate_sha256(file), 'features': self.__extract_features_from_binary(file, features_filename)})
        y = self.__extract_dataset_from_file(y_filename)
        return pd.DataFrame(X), y
    
    def load_test_train_from_file(self, X_filename, y_filename):
        """Loads an existing features vector and targets datasets

        Parameters
        ----------
        X_filename : str
            The file location of the features dataset, it must be in the form of csv or json
         y_filename : str
            The file location of the targets dataset, it must be in the form of csv or json
            
        Returns
        -------
        list<DataFrame>
            a list containing a dataframe that represents the extracted features vectors from binaries 
            and a dataframe representing targets variable
        """
        X = self.__parse_dataset_from_file(X_filename)
        y = self.__parse_dataset_from_file(y_filename)
        return X, y
    
    def __extract_features_from_binary(self, binary_name, features_filename=''):
        file_data = open(binary_name, 'rb').read()
        extractor = ember.features.PEFeatureExtractor(2, features_file=features_filename)
        features = np.array(extractor.feature_vector(file_data))
        return features

    def __extract_dataset_from_file(self, filename):
        df = None 
        ext = pathlib.Path(filename).suffix
        if(ext == '.json'):
            df = file_parser.JSONcreator().create_parser().parse(filename)
        elif(ext == '.csv'):
            df = file_parser.CSVcreator().create_parser().parse(filename)
            # decodes the list because is encoded in str
            if 'features' in df.columns:
                if df.features.dtype == 'object':
                    X = []
                    for idx, vec in enumerate(df['features']):
                        X.append(json.loads(vec))
                    df['features'] = X
        return df

    
def main():
    init_log('/home/students/derosa/prova', level=logging.DEBUG)
    #PEPreprocessingBuilder().from_existing_train_test_dataset('/home/students/derosa/test/', '/home/students/derosa/bodmas/bodmas_metadata.csv', '/home/students/derosa/config.json')
    pe1 = PEPreprocessingBuilder().new_train_test_dataset('/home/students/derosa/test/', '/home/students/derosa/bodmas/bodmas_metadata.csv', '/home/students/derosa/config.json')
    pe2 = PEPreprocessingBuilder().from_existing_raw_dataset('/home/students/derosa/bodmas/BODMAS/code/bodmas/prova.csv')
    X = pe2.data_cleansing().representation_transformation().get_X()
    X = pe2.data_cleansing().get_X()
    lgbm_model = lgb.Booster(model_file=MODEL_FOLDER + 'gbdt_bluehex_families_238_r10.txt')
    logging.info(predict(lgbm_model, 'gbdt', X, DATA_FOLDER + 'top_238_label_mapping.json'))
    pe1.data_cleansing().representation_transformation()
    #print(pe1.get_X_y())
    X, y = pe1.get_X_y()
    pe2.export_dataset('prova2.json')
    #pe2.data_cleansing().representation_transformation().export_dataset('X.csv', 'y.csv', protocol='csv')
    
    
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
    
    logging.debug(f'y_pred shape: {y_pred.shape} = {y_pred}')
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
