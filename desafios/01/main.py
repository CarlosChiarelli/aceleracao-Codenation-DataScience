#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[11]:


import pandas as pd
import numpy as np


# In[12]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:





# ## Análise exploratória

# In[13]:


black_friday.head()


# In[14]:


black_friday.info()


# In[15]:


explora = pd.DataFrame(data = {'colunas':list(black_friday.columns),
                    'tipos':list(black_friday.dtypes),
                    'na_perct':black_friday.isna().sum() / black_friday.shape[0],
                    'quantUnicos': black_friday.nunique()})

explora


# In[16]:


colsCont = list(explora[explora['quantUnicos'] < 10]['colunas'])


# In[17]:


print('CONTAGEM\n\n')
for coluna in colsCont:
    print(coluna, '\n', black_friday[coluna].value_counts() / black_friday.shape[0], end='\n\n')


# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[18]:


def q1():
    # Retorne aqui o resultado da questão 1.
    #return (black_friday.shape[0], black_friday.shape[1])
    return black_friday.shape


# In[ ]:





# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[20]:


black_friday.query("Gender == 'F' and Age == '26-35' ")['User_ID'].nunique()


# In[21]:


def q2():
    # Retorne aqui o resultado da questão 2.
    #black_friday.query("Gender == 'F' and Age == '26-35' ")['User_ID'].nunique()
    return black_friday.query("Gender == 'F' and Age == '26-35' ").shape[0]
    


# In[ ]:





# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[22]:


black_friday['User_ID'].nunique()


# In[23]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()
    


# In[ ]:





# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[24]:


explora['tipos'].nunique()
#explora['tipos'].value_counts()


# In[25]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return explora['tipos'].nunique()


# In[ ]:





# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[26]:


black_friday.isna().sum()


# In[27]:


type(float((black_friday.isna().sum(axis=1) > 0).mean()))


# In[28]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return float((black_friday.isna().sum(axis=1) > 0).mean())
    


# In[ ]:





# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[29]:


filtro = explora['na_perct'] == explora['na_perct'].max()
colMaisNull = explora[filtro]['colunas']
resp6 = black_friday[list(colMaisNull)].isna().sum() 
int(resp6)


# In[30]:


def q6():
    # Retorne aqui o resultado da questão 6.
    filtro = explora['na_perct'] == explora['na_perct'].max()
    colMaisNull = explora[filtro]['colunas']
    resp6 = black_friday[list(colMaisNull)].isna().sum() 
    return int(resp6)
    


# In[ ]:





# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[56]:


cont = black_friday['Product_Category_3'].value_counts()
#resp7 = cont.iloc[0]
resp7 = list(cont.index)[0]


# In[55]:


list(cont.index)[0]


# In[32]:


cont


# In[33]:


def q7():
    # Retorne aqui o resultado da questão 7.
    cont = black_friday['Product_Category_3'].value_counts()
    #resp7 = cont.iloc[0]
    resp7 = list(cont.index)[0]
    return resp7


# In[ ]:





# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[60]:


#black_friday['purchaseNormMedia'] = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
black_friday['purchaseNormMedia'] = (black_friday['Purchase'] - black_friday['Purchase'].min()) / (black_friday['Purchase'].max()-black_friday['Purchase'].min())
resp8 = black_friday['purchaseNormMedia'].mean()
float(resp8)


# In[35]:


black_friday.head()


# In[36]:


black_friday.describe()


# In[37]:


def q8():
    # Retorne aqui o resultado da questão 8.
    #black_friday['purchaseNormMedia'] = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
    resp8 = black_friday['purchaseNormMedia'].mean()
    return float(resp8)


# In[ ]:





# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[38]:


black_friday['purchaseNormMedia2'] = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
filtro = (black_friday['purchaseNormMedia2'] <= 1) & (black_friday['purchaseNormMedia2'] >= -1)  
resp9 = black_friday[filtro].shape[0]


# In[58]:


black_friday[filtro].describe()


# In[39]:


resp9


# In[40]:


def q9():
    # Retorne aqui o resultado da questão 9.
    black_friday['purchaseNormMedia2'] = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
    filtro = (black_friday['purchaseNormMedia2'] <= 1) & (black_friday['purchaseNormMedia2'] >= -1)  
    resp9 = black_friday[filtro].shape[0]
    return resp9


# In[ ]:





# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# questão 10.\n",
# 
#     "    compare = black_friday['Product_Category_2'].isnull() == black_friday['Product_Category_3'].isnull()\n",
# 
#     "    return (True in compare)"

# In[62]:


black_friday['Product_Category_2'].isnull() == black_friday['Product_Category_3'].isnull()


# In[41]:


aux = black_friday[['Product_Category_2','Product_Category_3']]
aux


# (bool(soma.sum() == 0))

# In[50]:


soma = aux['Product_Category_2'].isnull() != aux['Product_Category_3'].isnull()

if soma.sum() > 0:
    res10 = False
else:
    res10 = True


# In[51]:


res10


# In[44]:


def q10():
    # Retorne aqui o resultado da questão 10.
    aux = black_friday[['Product_Category_2','Product_Category_3']]
    soma = aux['Product_Category_2'].isnull() != aux['Product_Category_3'].isnull()
    
    if soma.sum() > 0:
        res10 = False
    else:
        res10 = True
        
    return bool(soma.sum() == 0)


# In[ ]:





# def printaTipos(lista):
#     for elem in lista:
#         print(elem)
#         print(type(elem), end='\n\n')

# printaTipos([q1(),q2(),q3(),q4(),q5(),q6(),q7(),q8(),q9(),q10()])

# In[ ]:





# In[ ]:





# In[ ]:




