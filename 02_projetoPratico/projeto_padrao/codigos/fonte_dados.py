# classe 1 
# aqui faço o split TREINO/TESTE

from pandas import read_csv
from sklearn.model_selection import train_test_split

class FonteDados:

    def __init__(self):
    	# criando variáveis
    	# definir os caminhos dados de treino e teste 
        self.caminho_dados = '../dados/train.csv'
        self.caminho_submissao = '../dados/test.csv'

    # definir se a etapa é dados de treino ou teste    
    def leitura_dados(self, dados_completos=False, etapa_treino=True, etapa_submissao=False):
        '''
            Lê os dados da fonte de dados
            :param dados_completos: booleano especificando se quero dados completos (sem split).
            :param etapa_treino: booleano especificando se é treino (falso é teste).
            :param etapa_submissao: booleano especificando se é etapa de submissão
            :return: pd.DataFrame com valores
        '''

        id_col = 'NU_INSCRICAO'
        target_col = 'IN_TREINEIRO'

        # selecionando colunas que existem na submissao
        df_sub = read_csv(self.caminho_submissao, index_col=id_col)
        
        cols_manter = df_sub.columns.tolist()
        cols_manter.append(target_col)

        if etapa_submissao:
            print('Retorno dados de submissão')
            print('Dimensão: ', df_sub.shape, '\n')
            return df_sub

        else:
            df = read_csv(self.caminho_dados, index_col=id_col)
            df = df[cols_manter]

            if dados_completos:
                print('Retorno dados completos')
                return df

            # split treino/teste
            # 80% para treino
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)
            # convertende de pd.Series para pd.DataFrame
            y_train = y_train.to_frame()
            y_test = y_test.to_frame()

            if etapa_treino:
                print('Retorno treino (X e y)')
                print('Dimensão X: ', X_train.shape, '\n','Dimensão y: ', y_train.shape, '\n')
                return X_train, y_train
            else:
                print('Retorno teste (X e y)')
                print('Dimensão X: ', X_test.shape, '\n','Dimensão y: ', y_test.shape, '\n')
                return X_test, y_test


