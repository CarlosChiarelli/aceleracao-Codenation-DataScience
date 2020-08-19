import pandas as pd
#from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


class Metricas:

    def __init__(self):
        pass

    def calcula_classif(self, y_true, y_pred):
        '''
        Calculate the metrics from a regression problem
        :param y_true: Numpy.ndarray or Pandas.Series
        :param y_pred: Numpy.ndarray or Pandas.Series
        :return: Dict with metrics
        '''
        #print('Cálculo das métricas')
        print(classification_report(y_true, y_pred), '\n\n')
        print('Matriz de confusão','\n', confusion_matrix(y_true, y_pred), '\n\n')  
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        print('Métrica f1:', f1.round(3),'\n\n')

        return {'f1':f1, 'recall':recall}
    