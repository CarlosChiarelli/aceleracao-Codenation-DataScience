#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns



class Visualizacao:

    def __init__(self):
        pass
        
    def variavel_viz(self, variavel):
        '''
        Visualizar uma variável do DF
        : param variavel: vetor/coluna do DF
        : return: 
        '''
        from funcoesProprias import plotaDistribuicaoUniVar

        print('Visualização uma variável (NAs removidos)')
        plotaDistribuicaoUniVar(variavel.dropna())

    def missings_viz(self, df, visualizar=True, escolhido_tipo=None, df_missings=False):
        '''
        Visualizar os missings, plota o tipo de visualizacao
        : param df: pd.DataFrame para visualizar
        : param visualizar: booleano para decidir qual visualizar
        : param escolhido_tipo: inteiro para decidir qual tipo visualizar
        : param df_missings: booleano para retorna Dataframe com percentual de nulos
        : return: pd.DataFrame com nomes das colunas e porcentagem missings
        '''

        if visualizar:
            # para quem usar um tema dark na IDE
            from matplotlib.pyplot import style
            style.use('classic')

            # colunas com missings apenas
            cols_miss = df.isnull().any()
            cols_miss = df.columns[cols_miss]

            if escolhido_tipo == None:
                print('Tipo de visualizacao: ', '\n', 'total de missings - 1',
                    '\n', 'ordem de aparição - 2', '\n', 'correlação - 3', 
                    '\n', 'dendograma - 4'
                    )
                escolhido_tipo = int(input())

            print('Visualização missings')
            # total
            if escolhido_tipo == 1:
                from missingno import bar
                bar(df[cols_miss])
            # ordem aparicao
            elif escolhido_tipo == 2:
                from missingno import matrix
                matrix(df[cols_miss])
            # correlacao
            elif escolhido_tipo == 3:
                from missingno import heatmap
                heatmap(df[cols_miss])
            # dendograma
            elif escolhido_tipo == 4:
                from missingno import dendrogram
                dendrogram(df[cols_miss])

        if df_missings:
            from funcoesProprias import dfExploracao

            print('Cálculo do percentual de missings num DataFrame')
            explora = dfExploracao(df)
            explora = explora.sort_values(['tipos','na_perct','quantUnicos'])
            return explora

    def correlacao_viz(self, df, colunas=None, anotado=False):
        '''
        Matriz de correlação de um DataFrame
        : param df: Dataframe
        : param colunas: lista de colunas a visualizar
        : param anotado: booleano para anotar valor da correlacao
        '''
        from seaborn import heatmap
        print('Visualizando correlação de Pearson (NAs removidos)')
        heatmap(df[colunas].dropna().corr(), annot=anotado)



    def regression_viz(self, y_true, y_pred, nome):
        '''
        Visualize the quality of regression model
        :param y_true: pd.Series with true label values
        :param y_pred: pd.Series with predicted label values
        :param nome: Name of the file wich will be saved
        :return: Save files in specified path
        '''
        residual = y_pred - y_true
        data = pd.DataFrame({'pred' : y_pred, 'true' : y_true, 'residual': residual})
        plot1 = sns.distplot(data['residual'], bins = 50)
        plot2 = sns.scatterplot(x= 'true', y = 'residual', data = data)
        plt.savefig(plot1, '../data/'+nome+'_distplot.csv')
        plt.savefig(plot1, '../data/' + nome + 'scatterplot.csv')
        plt.show()