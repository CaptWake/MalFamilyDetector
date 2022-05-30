import preprocessing
import lightgbm as lgb
import logging
import numpy as np
import json

# change this values to test the module

DATA_FOLDER = 'example/data/'
MODEL_FOLDER = 'example/model/'

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    
    pe1 = preprocessing.PEPreprocessingBuilder().from_existing_raw_dataset(DATA_FOLDER + 'prova.csv')
    X_1 = pe1.data_cleansing().representation_transformation().get_X()
    lgbm_model_1 = lgb.Booster(model_file=MODEL_FOLDER + 'gbdt_bluehex_families_238_r10.txt')
    logging.info(f"Predicted labels using model 1 -> {predict(lgbm_model_1, 'gbdt', X_1, DATA_FOLDER + 'top_238_label_mapping.json')}")
    
    # To run this test change the folder param of new_raw_dataset() method to a folder containing PE binaries 
    pe2 = preprocessing.PEPreprocessingBuilder().new_raw_dataset('binaries/', DATA_FOLDER + 'config.json')
    X_2 = pe2.data_cleansing().representation_transformation().get_X()
    lgbm_model_2 = lgb.Booster(model_file=MODEL_FOLDER + 'gbdt_bluehex_families_238_r10.txt')
    logging.info(f"Predicted labels using model 2 -> {predict(lgbm_model_2, 'gbdt', X_2, DATA_FOLDER + 'top_238_label_mapping.json')}")
    
    pe3 = preprocessing.PEPreprocessingBuilder().from_existing_train_test_dataset(DATA_FOLDER + 'X.csv', DATA_FOLDER + 'bodmas_metadata.csv')
    pe3.data_cleansing().representation_transformation().export_dataset('X.csv', 'y.csv')
    pe3.export_dataset('X.json', 'y.json', protocol='json')

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
    mapping = json.load(open(label_mapping_file, 'r'))
    return [mapping[str(prediction.item(0))] for prediction in y_pred]

if __name__ == '__main__':
    main()
