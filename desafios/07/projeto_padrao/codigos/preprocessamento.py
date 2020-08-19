# demanda muito tempo
# são muitos testes para ver ganho de performance 
# nas métricas do modelo 

from funcoesProprias import dfExploracao
#import category_encoders as ce
#import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from pandas import Series

class Preprocessamento:

    def __init__(self):
        # criando objetos que quero salvar com a classe
        # para aproveitá-los depois
        self.df_nomes_tipos_treino = None
        #self.feature_names = None
        #self.std_scaler = None
        #self.categoric_features = None
        #self.numeric_features = None
        #self.catb = None
        #self.scaler = None
        #self.train_features = None
        self.perc_miss_rm = None
        self.cols_rem = None
        self.cols_alteradas = None
        self.categ_ordinal = []
        self.vars_categ_ord = None
        self.vars_numericas = None
        self.imputador_miss = None
        self.imputador_miss_Y = None
        self.normalizador = None


    def processo(self, df_input, etapa_treino=True, perc_miss=.5, target=False):
        '''
        Processo para treinamento do modelo
        1. Discretiza e corrige variaveis: tipo_escola, Q025, sexo, ano_de_conclusao, raça
        2. Remove colunas por completude e significado
        3. Rotula categóricas ordinais

        :param df: Pandas DataFrame
        :param etapa_treino: Boolean
        :return: Pandas Data Frame processado

        '''

        # diferenciar etapa TREINO/TESTE
        # pois fit_transform Treino e apenas transform no Teste

        df = df_input.copy()

        # se quero processar o Y (target)
        if target:
            if etapa_treino:
                print('Preenchendo missings Y treino')
                self.imputador_miss_Y = SimpleImputer(strategy='constant', fill_value=0)
                df = self.imputador_miss_Y.fit_transform(df)
            else:
                print('Preenchendo missings Y teste', '\n\n')
                df = self.imputador_miss_Y.transform(df)

            return df

   
        def corrigeTipoEscola(x):
            if x == 1 or x == 2:
                return 0
            else:
                return 1

        def corrige_q025(x):
            if x == 'A':
                return 0
            else:
                return 1

        def corrige_sexo(x):
            if x == 'M':
                return 0
            else:
                return 1

        def categ_anoConclusao(x):
            # nao respondeu
            if x == 0:
                return 1
            # 2012 <= x <= 2015
            elif x >= 1 and x <= 4:
                return 2
            # 2012 > x
            else:
                return 3

        def raca_bin(x):
            if x == 1 or x == 4 or x == 0 or x == 6:
                return 0
            else:
                return 1

        print('Discretizando colunas')

        df['TP_ESCOLA'] =       df['TP_ESCOLA'].apply(corrigeTipoEscola)
        df['Q025'] =            df['Q025'].apply(corrige_q025)
        df['TP_SEXO'] =         df['TP_SEXO'].apply(corrige_sexo)
        df['TP_ANO_CONCLUIU'] = df['TP_ANO_CONCLUIU'].apply(categ_anoConclusao)
        df['TP_COR_RACA'] =     df['TP_COR_RACA'].apply(raca_bin) 

        self.cols_alteradas = ['TP_ESCOLA', 'Q025', 'TP_SEXO', 'TP_ANO_CONCLUIU', 'TP_COR_RACA']
        print('Colunas alteradas: tipo_escola, Q025, sexo, ano_de_conclusao, raça', '\n')

        if etapa_treino:
            # CALCULO tudo se for etapa de TREINO

            print('Definindo colunas a serem removidas')
            # colunas com muitos NAs
            self.perc_miss_rm = perc_miss
            cols_miss = df.isnull().mean() >= self.perc_miss_rm
            cols_miss = df.columns[cols_miss].tolist()

            # colunas irrelevantes
            cols_sem_relevancia = [True if x.startswith('CO_') or x.startswith('SG') or x.startswith('IN_') else False for x in df.columns]
            cols_sem_relevancia = df.columns[cols_sem_relevancia].tolist()
            
            cols_presenca = [True if 'PRESENCA' in x else False for x in df.columns]
            cols_presenca = df.columns[cols_presenca].tolist()

            # colunas para remover
            self.cols_rem = cols_miss + cols_sem_relevancia + cols_presenca + ['TP_NACIONALIDADE']

            print('Variáveis removidas (significância e completude', self.perc_miss_rm*100, '%)', '\n')
            df.drop(self.cols_rem, axis=1, inplace=True)

            # salvando colunas e tipos do treino
            print('Salvando tipos e nomes das colunas de treino', '\n')
            self.df_nomes_tipos_treino = dfExploracao(df)[['colunas', 'tipos']]

            # tipos categóricas ordinais
            self.vars_categ_ord = self.df_nomes_tipos_treino[self.df_nomes_tipos_treino['tipos'] == 'object']['colunas']
            self.vars_numericas = self.df_nomes_tipos_treino[self.df_nomes_tipos_treino['tipos'] != 'object']['colunas']

            print('Rotulação das categóricas ordinais', '\n')
            for coluna in self.vars_categ_ord:
                rotula_temp = LabelEncoder()
                df[coluna] = rotula_temp.fit_transform(df[coluna])
                self.categ_ordinal.append(rotula_temp)

            print('Preenchimento dos missings das numéricas', '\n')
            self.imputador_miss = SimpleImputer(strategy='constant', fill_value=0)
            df[self.vars_numericas] = self.imputador_miss.fit_transform(df[self.vars_numericas])

            print('Normalização dos dados (robusto)', '\n')
            self.normalizador = RobustScaler()
            df[self.vars_numericas] = self.normalizador.fit_transform(df[self.vars_numericas])


            # testar uma outra hora (codificação em variáveis dummies)
            # processar categoricas (catBoost um dos melhores para categoricas)
            #self.catb = ce.CatBoostEncoder(cols=self.categoric_features)
            #df[self.categoric_features] = self.catb.fit_transform(df[self.categoric_features], y=y)

            #return df[self.categoric_features + self.numeric_features], y
        
        else:
            # APLICO tudo se for etapa de TESTE

            print('Variáveis removidas (significância e completude ', self.perc_miss_rm*100, '%)', '\n')
            df.drop(self.cols_rem, axis=1, inplace=True)

            print('Rotulação das categóricas ordinais', '\n')
            for coluna, rotulador in zip(self.vars_categ_ord, self.categ_ordinal):
                df[coluna] = rotulador.transform(df[coluna])

            print('Preenchimento dos missings das numéricas', '\n')
            df[self.vars_numericas] = self.imputador_miss.transform(df[self.vars_numericas])

            print('Normalização dos dados (robusto)', '\n')
            df[self.vars_numericas] = self.normalizador.transform(df[self.vars_numericas])

            #df[self.categoric_features] = self.catb.transform(df[self.categoric_features])

            #return df[self.categoric_features + self.numeric_features]
        
        return df