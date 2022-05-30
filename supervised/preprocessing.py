import pandas as pd
import numpy as np
import ember

import file_parser
import pathlib
import hashlib
import json
import logging

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class PEPreprocessingBuilder:
    """
    This class allows the user to create the correct preprocessing interface 
    based on his particular needs.
    """
    def from_existing_train_test_dataset(self, X_fname, y_fname):
        """Creates a preprocessing interface containing the existing features and targets datasets

        Parameters
        ----------
        X_fname : str
            The file location of the features dataset
        y_fname : str
            The file location of the targets dataset

        Returns
        -------
        TrainTestPEPreprocessing
            a preprocessing interface used to generate the pre-processed datasets from the loaded datasets
        """
        X, y = PEDataset().load_test_train_from_file(X_fname, y_fname)
        logging.info(f'Loaded successfully features dataset {X_fname} and targets dataset {y_fname} ...')
        tpe = TrainTestPEPreprocessing()
        tpe.set_dataset({'X': X, 'y': y})
        return tpe

    def new_train_test_dataset(self, dirname,  y_fname, features_fname):
        """Creates a preprocessing interface containing the existing targets dataset and the newly generated features dataset 

        Parameters
        ----------
        dirname : str
            The directory location of the samples used for building the features dataset
        y_fname : str
            The file location of the targets dataset
        features_fname : str
            The file location of the configuration file containing the features to be extracted from the samples

        Returns
        -------
        TrainTestPEPreprocessing
            a preprocessing interface used to generate the pre-processed datasets from the loaded datasets
        """
        X, y = PEDataset().generate_test_train(dirname, y_fname, features_fname)
        logging.info('Generated successfully features dataset ...')
        logging.info(f'Loaded successfully targets dataset {y_fname} ...')
        tpe = TrainTestPEPreprocessing()
        tpe.set_dataset({'X': X, 'y': y})
        return tpe

    def from_existing_raw_dataset(self, fname):
        """Creates a preprocessing interface containing the existing features dataset

        Parameters
        ----------
        fname : str
            The file location of the dataset

        Returns
        -------
        RawPEPreprocessing
            a preprocessing interface used to generate the pre-processed dataset from the loaded dataset
        """
        df = PEDataset().load_raw_from_file(fname)
        logging.info(f'Loaded successfully features dataset {fname}...')
        rpe = RawPEPreprocessing()
        rpe.set_dataset(df)
        return rpe

    def new_raw_dataset(self, dirname, features_fname):
        """Creates a preprocessing interface containing the newly generated features dataset

        Parameters
        ----------
        dirname : str
            The directory location of the samples used for building the features dataset
        features_fname : str
            The file location of the configuration file containing the features to be extracted from the samples

        Returns
        -------
        RawPEPreprocessing
            a preprocessing interface used to generate the pre-processed dataset from the loaded dataset
        """
        df = PEDataset().generate_raw(dirname, features_fname)
        logging.info('Generated successfully features dataset ...')
        rpe = RawPEPreprocessing()
        rpe.set_dataset(df)
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
        # select only feature columns
        return self.dataset.iloc[:, 0:-1].to_numpy()
    
    def export_dataset(self, X_filename, protocol='csv'):
        if protocol == 'csv':
            self.dataset.to_csv(X_filename, index=False)
        exported = self.dataset.to_json(orient="records")
        with open(X_filename, 'w+') as f:
            json.dump(json.loads(exported), f)
    
    def data_cleansing(self):
        self.dataset.dropna(axis=0)
        logging.debug('Successfully removed missing values ...')
        self.dataset.drop_duplicates(subset=['sha256'], inplace=True)
        logging.debug('Successfully removed duplicates  ...')
        return self

    def representation_transformation(self):
        # select only feature columns
        self.dataset.iloc[:, 0:-1] = self._normalize(self.get_X())
        logging.debug('Successfully normalized features vector values ...')
        return self

    
class TrainTestPEPreprocessing(PEPreprocessing):
    """
    A preprocessing interface focussed on manipulating the features and targets datasets in order
    to obatin the pre-processed datasets that'll be used to feed the classificator 
    for training and testing purposes
    """
    def get_X_y(self):
        return self.dataset['X'].iloc[:, 0:-1].to_numpy(), self.dataset['y']
    
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
        logging.debug('Successfully removed missing values ...')
        self.dataset['X'].drop_duplicates(subset=['sha256'], inplace=True)
        self.dataset['y'].drop_duplicates(subset=['sha'], inplace=True)
        self.dataset['y'] = self.dataset['y'][(self.dataset['y']['sha'].isin(self.dataset['X']['sha256']))]
        logging.debug('Successfully removed duplicates  ...')
        return self
    
    def representation_transformation(self):
        X, y = self.get_X_y()
        self.dataset['X'].iloc[:, 0:-1] = self._normalize(X)
        logging.debug('Successfully normalized features vector values ...')
        self.dataset['y']['family'] = self.__prepare_targets(y['family'].values)
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
        """Creates a feature vectors dataframe from raw binaries inside a folder

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
        X, hashes = [], []
        for file in pathlib.Path(dir_name).glob('*'):
            X.append(self.__extract_features_from_binary(file, features_filename))
            hashes.append(calculate_sha256(file))
        X = np.array(X)
        feat_cols = ['feat'+str(i) for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feat_cols)
        df['sha256'] = hashes
        return df
    
    def load_raw_from_file(self, X_fname):
        """Loads a feature vectors dataframe

        Parameters
        ----------
        X_fname : str
            The file location of the features dataset, it must be in the form of csv or json

        Returns
        -------
        DataFrame
            a dataframe containing the loaded features dataset
        """
        return self.__extract_dataset_from_file(X_fname)
    
    def generate_test_train(self, dirname, y_fname, features_fname):
        """"Creates a features vector dataframe from raw binaries inside a folder

        Parameters
        ----------
        dirname : str
            The folder location containing the binaries
        y_fname : str
            The file location of the targets dataset, it must be in the form of csv or json
        features_fname: str
            The file location used for feature selection
        
        Returns
        -------
        list<DataFrame>
            a list containing a dataframe that represents the extracted features vectors from binaries 
            and a dataframe representing targets variable
        """
        X = self.generate_raw(dirname, features_fname)
        y = self.__extract_dataset_from_file(y_fname)
        return X, y
    
    def load_test_train_from_file(self, X_fname, y_fname):
        """Loads an existing features vector and targets datasets

        Parameters
        ----------
        X_fname : str
            The file location of the features dataset, it must be in the form of csv or json
         y_fname : str
            The file location of the targets dataset, it must be in the form of csv or json
            
        Returns
        -------
        list<DataFrame>
            a list containing a dataframe that represents the extracted features vectors from binaries 
            and a dataframe representing targets variable
        """
        X = self.__extract_dataset_from_file(X_fname)
        y = self.__extract_dataset_from_file(y_fname)
        return X, y
    
    def __extract_features_from_binary(self, binary_name, features_filename=''):
        file_data = open(binary_name, 'rb').read()
        extractor = ember.features.PEFeatureExtractor(2, features_file=features_filename)
        return np.array(extractor.feature_vector(file_data))

    def __extract_dataset_from_file(self, filename):
        #df = None 
        ext = pathlib.Path(filename).suffix
        if(ext == '.json'):
            df = file_parser.JSONcreator().create_parser().parse(filename)
        elif(ext == '.csv'):
            df = file_parser.CSVcreator().create_parser().parse(filename)
        return df

def calculate_sha256(filename):
    sha256_hash = hashlib.sha256()
    with open(filename,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
        return(sha256_hash.hexdigest())
