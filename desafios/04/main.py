#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[64]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
#import funcoesProprias as fp
from statsmodels.api import qqplot


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize

figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head()


# In[6]:


athletes.tail()


# In[10]:


athletes.shape


# In[9]:


athletes.info()


# In[7]:


athletes.describe()


# explora = fp.dfExploracao(athletes)
# explora

# In[15]:


athletes.bronze.unique()


# # #[col if col in ['id','name','dob'].join() for col in list(explora.colunas)]
# 
# aux=[]
# for x in explora.colunas:
#     if not x in ['id','name','dob']:
#         aux.append(True)
#     else:
#         aux.append(False)

# explora[aux]

# colsObj = aux & (explora.tipos=='object')
# colsObj = explora[colsObj].colunas
# 
# colsNum = (explora.colunas != 'id') & (explora.colunas != 'name') & (explora.tipos!='object')
# colsNum = explora[colsNum].colunas

# fp.plotNumVsNum(athletes[colsNum])

# athletes[colsNum].hist(bins=30, figsize=(20,15))

# fp.plotObjCols(athletes[colsObj])

# In[ ]:





# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[70]:


dadosNorm = get_sample(athletes, 'height', n=3000)

pValor = sct.shapiro(dadosNorm)[1]
if pValor < .05:
    print('pValor: ', pValor,'\nnão é normal!')


# In[67]:


qqplot(dadosNorm, fit=True, line='45')


# In[71]:


def q1():
    # Retorne aqui o resultado da questão 1.
    dadosNorm = get_sample(athletes, 'height', n=3000)

    pValor = sct.shapiro(dadosNorm)[1]
    if pValor < .05:
        print('pValor: ', pValor,'\nnão é normal!')
        return False
    else:
        return True


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[86]:


sns.distplot(athletes['height'], bins=25)


# In[87]:


sns.boxplot(athletes['height'])


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[76]:


pValor = sct.jarque_bera(dadosNorm)[1]

if pValor < .05:
    print('pValor: ',pValor,'\nnão é normal')


# In[78]:


def q2():
    # Retorne aqui o resultado da questão 2.
    dadosNorm = get_sample(athletes, 'height', n=3000)
    pValor = sct.jarque_bera(dadosNorm)[1]

    if pValor < .05:
        print('pValor: ',pValor,'\nnão é normal')
        return False
    else: 
        return True


# __Para refletir__:
# 
# * Esse resultado faz sentido?
# 
# Sim, pois a assimetria e a curtose lembram uma distribuição normal, logo o p-valor não foi tão pequeno como no Shapiro, diminuindo a certeza de rejeição de H0.

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[88]:


dadosNorm = get_sample(athletes, 'weight', n=3000)

sct.normaltest(dadosNorm).pvalue


# In[89]:


def q3():
    # Retorne aqui o resultado da questão 3.
    dadosNorm = get_sample(athletes, 'weight', n=3000)
    
    if sct.normaltest(dadosNorm).pvalue < .05:
        return False
    else: 
        return True


# In[90]:


q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# 
# Sim, a cauda não parece de uma normal.
# 
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[81]:


sns.distplot(athletes['weight'], bins=25)


# In[85]:


sns.boxplot(athletes['weight'])


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[92]:


dadosNorm = np.log(get_sample(athletes, 'weight', n=3000))

sct.normaltest(dadosNorm).pvalue


# In[93]:


def q4():
    # Retorne aqui o resultado da questão 4.
    dadosNorm = np.log(get_sample(athletes, 'weight', n=3000))
    
    if sct.normaltest(dadosNorm).pvalue < .05:
        return False
    else:
        return True


# In[97]:


q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# 
# Sim, se aproximou de uma normal.
# 
# * Você esperava um resultado diferente agora?
# 
# Sim.

# In[95]:


sns.distplot(np.log(athletes['weight']), bins=25)


# In[96]:


sns.boxplot(np.log(athletes['weight']))


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[112]:


filtro = athletes['nationality'].str.contains('BRA|USA|CAN')
aux = athletes[filtro]

# bra
bra = aux[aux.nationality.str.contains('BRA')]
usa = aux[aux.nationality.str.contains('USA')]


# In[108]:


print('dimensoes',bra.shape, usa.shape)
print('variancias',bra.height.std()**2, usa.height.std()**2)
print('media',bra.height.mean(), usa.height.mean())


# In[124]:


pValor = sct.ttest_ind(bra.height, usa.height, equal_var=False, nan_policy='omit').pvalue
print('pValor: ',pValor)

if pValor < .05:
    print('as médias não são iguais!')
else:
    print('as médias são iguais')


# In[129]:


sns.distplot(bra.height, bins=30)
sns.distplot(usa.height, bins=30)


# In[ ]:





# In[125]:


def q5():
    # Retorne aqui o resultado da questão 5.
    filtro = athletes['nationality'].str.contains('BRA|USA|CAN')
    aux = athletes[filtro]

    # bra
    bra = aux[aux.nationality.str.contains('BRA')]
    usa = aux[aux.nationality.str.contains('USA')]

    pValor = sct.ttest_ind(bra.height, usa.height, equal_var=False, nan_policy='omit').pvalue
    print('pValor: ',pValor)

    if pValor < .05:
        print('as médias não são iguais!')
        return False
    else:
        print('as médias são iguais')
        return True
    


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[130]:


filtro = athletes['nationality'].str.contains('BRA|USA|CAN')
aux = athletes[filtro]

# bra
bra = aux[aux.nationality.str.contains('BRA')]
can = aux[aux.nationality.str.contains('CAN')]

print('dimensoes',bra.shape, can.shape)
print('variancias',bra.height.std()**2, can.height.std()**2)
print('media',bra.height.mean(), can.height.mean())

pValor = sct.ttest_ind(bra.height, can.height, equal_var=False, nan_policy='omit').pvalue
print('pValor: ',pValor)

if pValor < .05:
    print('as médias não são iguais!')
else:
    print('as médias são iguais')


# In[131]:


sns.distplot(bra.height, bins=30)
sns.distplot(can.height, bins=30)


# In[132]:


def q6():
    filtro = athletes['nationality'].str.contains('BRA|USA|CAN')
    aux = athletes[filtro]

    # bra
    bra = aux[aux.nationality.str.contains('BRA')]
    can = aux[aux.nationality.str.contains('CAN')]

    print('dimensoes',bra.shape, can.shape)
    print('variancias',bra.height.std()**2, can.height.std()**2)
    print('media',bra.height.mean(), can.height.mean())

    pValor = sct.ttest_ind(bra.height, can.height, equal_var=False, nan_policy='omit').pvalue
    print('pValor: ',pValor)

    if pValor < .05:
        print('as médias não são iguais!')
        return False
    else:
        print('as médias são iguais')
        return True


# In[133]:


q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[140]:


filtro = athletes['nationality'].str.contains('BRA|USA|CAN')
aux = athletes[filtro]

# bra
usa = aux[aux.nationality.str.contains('USA')]
can = aux[aux.nationality.str.contains('CAN')]

print('dimensoes',usa.shape, can.shape)
print('variancias',usa.height.std()**2, can.height.std()**2)
print('media',usa.height.mean(), can.height.mean())

pValor = sct.ttest_ind(usa.height, can.height, equal_var=False, nan_policy='omit').pvalue
print('pValor: ',pValor)

if pValor < .05:
    print('as médias não são iguais!')
    #return False
else:
    print('as médias são iguais')
    #return True


# In[141]:


def q7():
    filtro = athletes['nationality'].str.contains('BRA|USA|CAN')
    aux = athletes[filtro]

    # bra
    usa = aux[aux.nationality.str.contains('USA')]
    can = aux[aux.nationality.str.contains('CAN')]

    print('dimensoes',usa.shape, can.shape)
    print('variancias',usa.height.std()**2, can.height.std()**2)
    print('media',usa.height.mean(), can.height.mean())

    pValor = sct.ttest_ind(usa.height, can.height, equal_var=False, nan_policy='omit').pvalue
    print('pValor: ',pValor)

    return float(round(pValor, 8))


# In[143]:


q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
