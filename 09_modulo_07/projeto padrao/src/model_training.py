# classe 2

# parte final !!!
# sem experimentos

# uso essa classe quando já tiver o modelo consistente (modelo final)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load

from data_source import DataSource
from preprocessing import Preprocessing
from experiments import Experiments 


class ModelTraining:

    # começo aqui chamando dataSource e preProcessamento
    # como preProc não foi definido então 
    def __init__(self):
        self.data = DataSource()
        self.preprocessing = None
        
    def model_training(self):
        '''
        Train the model.
        :return: Dict with trained model, preprocessing used and columns used in training
        '''

        # chamo o prePocessamento
        pre = Preprocessing()

        # leio os dados
        print('Loading data')
        df = self.data.read_data(etapa_treino = True)

        # preProcessamento
        print('Training preprocessing')
        # para treino
        X_train, y_train = pre.process(df, etapa_treino = True)


        print('Training Model')
        # chamo uma regLinear mas já poderia linkar
        # com a classe Experiment e retorna o experimento com 
        # a melhor métrica
        model_obj = LinearRegression()
        model_obj.fit(X_train, y_train)

        # guardando informacoes no dicionario
        model = {'model_obj' : model_obj,
                 'preprocessing' : pre,
                 'colunas' : pre.feature_names }
        print(model)

        # salvando modelo treinado com informacoes
        dump(model, '../output/modelo.pkl')

        # retorna o dicionario de modelo
        return model
    
    