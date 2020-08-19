# classe 2

# parte final !!!
# sem experimentos

# uso essa classe quando já tiver o modelo consistente (modelo final)

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from joblib import dump, load

from fonte_dados import FonteDados
from preprocessamento import Preprocessamento
from experimentos import Experimentos 


class TreinamentoModelo:

    # começo aqui chamando dataSource e preProcessamento
    # como preProc não foi definido então 
    def __init__(self):
        self.dados = FonteDados()
        self.pre_proc = None
        
    def treinamento_modelo(self):
        '''
        Train the model.
        :return: Dict with trained model, preprocessing used and columns used in training
        '''
        from numpy.random import seed
        
        # chamo o prePocessamento
        self.pre_proc = Preprocessamento()

        # leio os dados
        print('Carregamento dos dados', '\n\n')
        X_treino, y_treino = self.dados.leitura_dados()
        #df = self.dados.read_data(etapa_treino = True)

        # preProcessamento
        print('Treinamento do pré-processamento', '\n\n')
        # para treino
        X_treino = self.pre_proc.processo(X_treino)
        #y_treino = self.pre_proc.processo(y_treino, target=True)
        #X_train, y_train = pre.process(df, etapa_treino = True)

        print('Balanceamento Oversampling', '\n\n')
        #self.y_treino = y_treino
        X_treino, y_treino = self.pre_proc.balanceamento_oversampling(X_treino, y_treino)

        print('Treinamento do modelo', '\n\n')
        # chamo uma regLinear mas já poderia linkar
        # com a classe Experiment e retorna o experimento com 
        # a melhor métrica
        seed(42)
        model_obj = XGBClassifier()
        model_obj.fit(X_treino, y_treino)

        # guardando informacoes no dicionario
        model = {'model_obj' : model_obj,
                 'preprocess' : self.pre_proc,
                 'colunas' : self.pre_proc.df_nomes_tipos_treino }
        print(model)

        # salvando modelo treinado com informacoes
        dump(model, '../saida/modelo.pkl')

        # retorna o dicionario de modelo
        return model
    
    