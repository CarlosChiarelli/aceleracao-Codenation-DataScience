# classe responsável por fazer as previsões

import pandas as pd
import numpy as np
from joblib import dump, load

from model_training import ModelTraining
from metrics import Metrics
from preprocessing import Preprocessing
from data_source import DataSource

class ModelInference:
    def __init__(self):
        self.modelo = None

    def predict(self):
        '''
        Predict values using model trained.
        :return: pd.Series with predicted values.
        '''

        # carregar o modelo treinado
        print('Loading the model')
        self.modelo = load('../output/modelo.pkl')

        # ler os dados de TESTE 
        print('Loading Data')
        test_df, y_test = DataSource().read_data(etapa_treino=False)
        
        print('Preprocessing Data')
        X_test = self.modelo['preprocessing'].process(test_df, etapa_treino=False)
        print(X_test.isna().sum())

        print('Predicting')
        y_pred = self.modelo['model_obj'].predict(X_test)

        # salvando resultado predito
        print('Saving Files')
        pd.DataFrame(y_pred).to_csv('../output/predito.csv')

        return y_pred
