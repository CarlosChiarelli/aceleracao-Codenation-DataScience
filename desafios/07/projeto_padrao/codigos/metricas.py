import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

class Metricas:

    def __init__(self):
        pass

    def calcula_regressao(self, y_true, y_pred):
        '''
        Calculate the metrics from a regression problem
        :param y_true: Numpy.ndarray or Pandas.Series
        :param y_pred: Numpy.ndarray or Pandas.Series
        :return: Dict with metrics
        '''
        #print('Cálculo das métricas')
        erro_medio_abs = mean_absolute_error(y_true, y_pred).round(2)
        erro_medio_quad = mean_squared_error(y_true, y_pred).round(2)
        r2 = r2_score(y_true, y_pred).round(2)
        
        return {'erro_medio_abs' : erro_medio_abs, 'erro_medio_quad' : erro_medio_quad, 'r2' : r2}
    